//! Integration tests for the `checkConstraints` writer feature (prototype, gated by the
//! `check-constraints-in-dev` cargo feature).

use std::sync::Arc;

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::Snapshot;
use test_utils::engine_store_setup;

fn test_schema() -> Arc<StructType> {
    Arc::new(StructType::new_unchecked([
        StructField::nullable("amount", DataType::LONG),
        StructField::nullable("name", DataType::STRING),
    ]))
}

#[tokio::test]
#[cfg(not(feature = "check-constraints-in-dev"))]
async fn write_blocked_when_cargo_feature_off() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;

    use test_utils::create_table_with_configuration;

    let (store, engine, table_location) = engine_store_setup("test_cc_off", None);
    let table_url = create_table_with_configuration(
        store,
        table_location,
        test_schema(),
        &[],
        true,
        vec![],
        vec!["checkConstraints"],
        HashMap::from([(
            "delta.constraints.positive_amount".to_string(),
            "amount > 0".to_string(),
        )]),
    )
    .await?;

    let snapshot = Snapshot::builder_for(table_url).build(&engine)?;
    let err = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), &engine)
        .expect_err("write must be blocked when checkConstraints is unsupported");
    assert!(
        err.to_string().contains("checkConstraints"),
        "error must name the unsupported feature; got: {err}",
    );
    Ok(())
}

#[cfg(feature = "check-constraints-in-dev")]
mod enabled {
    use std::collections::HashMap;

    use delta_kernel::arrow::array::{ArrayRef, Int64Array, StringArray};
    use delta_kernel::arrow::record_batch::RecordBatch;
    use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
    use delta_kernel::engine::arrow_data::ArrowEngineData;
    use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
    use delta_kernel::engine::default::DefaultEngine;
    use delta_kernel::expressions::Scalar;
    use delta_kernel::object_store::DynObjectStore;
    use delta_kernel::transaction::Transaction;
    use delta_kernel::{Engine as _, Error};
    use itertools::Itertools as _;
    use test_utils::{create_table_with_configuration, test_read};
    use url::Url;

    use super::*;

    /// Creates an unpartitioned table with the given `delta.constraints.<name>` entries and
    /// returns `(table_url, engine)`.
    async fn setup_constrained_table(
        test_name: &str,
        constraints: &[(&str, &str)],
    ) -> Result<(Url, DefaultEngine<TokioBackgroundExecutor>), Box<dyn std::error::Error>> {
        setup_constrained_table_partitioned(test_name, constraints, &[]).await
    }

    /// Like [`setup_constrained_table`], with partition columns.
    async fn setup_constrained_table_partitioned(
        test_name: &str,
        constraints: &[(&str, &str)],
        partition_columns: &[&str],
    ) -> Result<(Url, DefaultEngine<TokioBackgroundExecutor>), Box<dyn std::error::Error>> {
        let (store, engine, table_location): (Arc<DynObjectStore>, _, _) =
            engine_store_setup(test_name, None);
        let configuration = constraints
            .iter()
            .map(|(name, sql)| (format!("delta.constraints.{name}"), sql.to_string()))
            .collect();
        let table_url = create_table_with_configuration(
            store,
            table_location,
            test_schema(),
            partition_columns,
            true,
            vec![],
            vec!["checkConstraints"],
            configuration,
        )
        .await?;
        Ok((table_url, engine))
    }

    fn begin_txn(
        table_url: &Url,
        engine: &DefaultEngine<TokioBackgroundExecutor>,
    ) -> Result<Transaction, Box<dyn std::error::Error>> {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(engine)?;
        Ok(snapshot.transaction(Box::new(FileSystemCommitter::new()), engine)?)
    }

    fn batch(
        amounts: Vec<Option<i64>>,
        names: Vec<&str>,
    ) -> Result<ArrowEngineData, Box<dyn std::error::Error>> {
        let arrow_schema = Arc::new(test_schema().as_ref().try_into_arrow()?);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(amounts)),
            Arc::new(StringArray::from(names)),
        ];
        Ok(ArrowEngineData::new(RecordBatch::try_new(
            arrow_schema,
            columns,
        )?))
    }

    fn assert_err_contains(err: Error, needle: &str) {
        let msg = err.to_string();
        assert!(msg.contains(needle), "expected '{needle}' in error: {msg}");
    }

    #[tokio::test]
    async fn write_context_creation_does_not_require_acknowledgment(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) =
            setup_constrained_table("test_cc_gate_wc", &[("positive_amount", "amount > 0")])
                .await?;
        // Acknowledgment now happens by calling check_constraints() -- which may come after the
        // write context is built -- so creating a write context no longer gates. The gate is at
        // commit (see commit_requires_acknowledgment_only_when_adding_data).
        let txn = begin_txn(&table_url, &engine)?;
        txn.unpartitioned_write_context()
            .expect("write-context creation must not require acknowledgment");
        Ok(())
    }

    #[tokio::test]
    async fn commit_requires_acknowledgment_only_when_adding_data(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) =
            setup_constrained_table("test_cc_gate_commit", &[("positive_amount", "amount > 0")])
                .await?;

        // Constraints apply only to added rows: a commit with no add files (e.g. a
        // metadata-only ALTER) needs no acknowledgment.
        let txn = begin_txn(&table_url, &engine)?.with_operation("WRITE".to_string());
        txn.commit(&engine)?.unwrap_committed();

        // But a connector that bypasses kernel write contexts and registers add files
        // directly is still stopped at commit.
        let mut txn = begin_txn(&table_url, &engine)?.with_operation("WRITE".to_string());
        let fabricated = test_utils::create_add_files_metadata(
            txn.add_files_schema(),
            vec![("part-00000.parquet", 1024, 1000000, Some(1))],
        )?;
        txn.add_files(fabricated);
        let err = txn
            .commit(&engine)
            .expect_err("commit with add files must require acknowledgment");
        assert_err_contains(err, "Transaction::check_constraints()");
        Ok(())
    }

    /// Approach A: a custom connector discovers constraints on the transaction and validates
    /// each batch itself before writing.
    #[tokio::test]
    async fn connector_driven_validate_catches_violations() -> Result<(), Box<dyn std::error::Error>>
    {
        let (table_url, engine) = setup_constrained_table(
            "test_cc_connector_driven",
            &[("positive_amount", "amount > 0")],
        )
        .await?;
        // Calling check_constraints() below is itself the acknowledgment (no separate opt-in).
        let txn = begin_txn(&table_url, &engine)?;

        // Set-level question first: kernel can handle everything on this table.
        let constraints = txn.check_constraints();
        assert!(constraints.is_kernel_parsable());

        let constraint = constraints
            .iter()
            .exactly_one()
            .expect("table has exactly one constraint");
        assert_eq!(constraint.name(), "positive_amount");
        assert_eq!(constraint.raw_sql(), "amount > 0");
        // Kernel parsed it; the batch-validation calls below prove it is data-batch enforced.
        assert!(constraint.predicate().is_some());

        let evaluation_handler = engine.evaluation_handler();

        // All rows satisfy the constraint.
        let good = batch(vec![Some(1), Some(5)], vec!["a", "b"])?;
        constraint.validate(&good, evaluation_handler.as_ref())?;

        // A false row violates.
        let bad = batch(vec![Some(1), Some(-5)], vec!["a", "b"])?;
        let err = constraint
            .validate(&bad, evaluation_handler.as_ref())
            .expect_err("negative amount must violate");
        assert_err_contains(err, "positive_amount");

        // A NULL predicate result also violates (protocol: only `true` passes).
        let null_row = batch(vec![Some(1), None], vec!["a", "b"])?;
        let err = constraint
            .validate(&null_row, evaluation_handler.as_ref())
            .expect_err("NULL amount must violate");
        assert_err_contains(err, "NULL");
        Ok(())
    }

    /// A DefaultEngine connector validates each batch explicitly (via the constraints' validator)
    /// before writing it -- `write_parquet` does not validate. A violating batch is caught, and a
    /// satisfying write commits and round-trips.
    #[tokio::test]
    async fn connector_validates_batch_before_write() -> Result<(), Box<dyn std::error::Error>> {
        // Spark stores parser-round-tripped, token-spaced expression text; use that style here.
        let (table_url, engine) = setup_constrained_table(
            "test_cc_validate_before_write",
            &[("positive_amount", "( amount > 0 )")],
        )
        .await?;
        let mut txn = begin_txn(&table_url, &engine)?.with_operation("WRITE".to_string());
        // Calling check_constraints() acknowledges; build a validator to enforce before writing.
        let constraints = txn.check_constraints();
        let handler = engine.evaluation_handler();
        let validator = constraints.validator(handler.as_ref())?;
        let write_context = txn.unpartitioned_write_context()?;

        // Violating batch: the connector's validation catches it before any file is written.
        let bad = batch(vec![Some(1), Some(-5)], vec!["a", "b"])?;
        let err = validator
            .validate(&bad)
            .expect_err("validation must reject a violating batch");
        assert_err_contains(err, "positive_amount");

        // Satisfying batch: validate, then write, commit, and read back.
        let good = batch(vec![Some(1), Some(5)], vec!["a", "b"])?;
        validator.validate(&good)?;
        let add_files_metadata = engine.write_parquet(&good, &write_context).await?;
        txn.add_files(add_files_metadata);
        txn.commit(&engine)?.unwrap_committed();

        let expected = batch(vec![Some(1), Some(5)], vec!["a", "b"])?;
        test_read(&expected, &table_url, Arc::new(engine))?;
        Ok(())
    }

    /// Constraints kernel cannot parse (here: a junction, outside the simple-comparison
    /// subset) are surfaced as connector-enforced, and building a validator over them fails closed.
    #[tokio::test]
    async fn non_parsable_constraint_fails_closed() -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) = setup_constrained_table(
            "test_cc_non_parsable",
            &[("amount_range", "amount > 0 AND amount < 100")],
        )
        .await?;
        // Calling check_constraints() below is itself the acknowledgment (no separate opt-in).
        let txn = begin_txn(&table_url, &engine)?;

        // The connector's first, set-level question: can kernel handle all the constraints?
        let constraints = txn.check_constraints();
        assert!(!constraints.is_kernel_parsable());

        // No: kernel does not hand back a connector-owned subset -- the connector iterates and
        // finds the constraint has no predicate, so only its raw SQL is available for it to
        // evaluate itself (a connector with no SQL engine must instead refuse to write).
        let raw: Vec<_> = constraints
            .iter()
            .filter(|c| c.predicate().is_none())
            .collect();
        let [raw] = raw[..] else {
            panic!("table has exactly one unparsable constraint");
        };
        assert_eq!(raw.raw_sql(), "amount > 0 AND amount < 100");

        // Kernel cannot evaluate it, so building a validator fails closed -- a DefaultEngine
        // connector with no SQL engine of its own must refuse to write.
        let handler = engine.evaluation_handler();
        let err = constraints
            .validator(handler.as_ref())
            .map(|_| ())
            .expect_err("non-parsable constraint must fail closed");
        assert_err_contains(err, "must enforce");
        Ok(())
    }

    /// Tables whose protocol lists `checkConstraints` but that define no constraints are
    /// writable without acknowledgment (nothing to enforce).
    #[tokio::test]
    async fn feature_without_constraints_needs_no_acknowledgment(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) = setup_constrained_table("test_cc_feature_only", &[]).await?;
        let mut txn = begin_txn(&table_url, &engine)?.with_operation("WRITE".to_string());
        assert!(txn.check_constraints().is_empty());

        let write_context = txn.unpartitioned_write_context()?;
        let data = batch(vec![Some(-1)], vec!["a"])?; // no constraints: any value is fine
        let add_files_metadata = engine.write_parquet(&data, &write_context).await?;
        txn.add_files(add_files_metadata);
        txn.commit(&engine)?.unwrap_committed();
        Ok(())
    }

    /// Constraints on a partition column are validated as ordinary predicates over the full
    /// (pre-partition) batch, which still carries the partition column as data -- no special
    /// partition handling. A data-column constraint on the same table validates the same way. A
    /// satisfying write commits and round-trips, with the partition column reconstructed from
    /// `add.partitionValues`.
    #[tokio::test]
    async fn partition_column_constraint_validates_over_full_batch(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) = setup_constrained_table_partitioned(
            "test_cc_partition_col",
            &[
                ("name_check", "name = 'a'"),
                ("positive_amount", "amount > 0"),
            ],
            &["name"],
        )
        .await?;
        let mut txn = begin_txn(&table_url, &engine)?.with_operation("WRITE".to_string());

        // Both constraints parse (the partition-column one is just a predicate).
        let constraints = txn.check_constraints();
        assert!(constraints.is_kernel_parsable());
        let handler = engine.evaluation_handler();
        let validator = constraints.validator(handler.as_ref())?;

        // A row violating the partition-column constraint (name != 'a') is caught.
        let err = validator
            .validate(&batch(vec![Some(1)], vec!["b"])?)
            .expect_err("name 'b' must violate name_check");
        assert_err_contains(err, "name_check");

        // A row violating the data-column constraint (amount <= 0) is caught too.
        let err = validator
            .validate(&batch(vec![Some(-5)], vec!["a"])?)
            .expect_err("negative amount must violate positive_amount");
        assert_err_contains(err, "positive_amount");

        // A satisfying batch validates; write it to the matching partition, commit, and read back.
        let good = batch(vec![Some(5)], vec!["a"])?;
        validator.validate(&good)?;
        let write_context = txn
            .partitioned_write_context(HashMap::from([("name".to_string(), Scalar::from("a"))]))?;
        let add_files_metadata = engine.write_parquet(&good, &write_context).await?;
        txn.add_files(add_files_metadata);
        txn.commit(&engine)?.unwrap_committed();

        let expected = batch(vec![Some(5)], vec!["a"])?;
        test_read(&expected, &table_url, Arc::new(engine))?;
        Ok(())
    }

    /// A constraint referencing BOTH a partition column and a data column (`region != label`) is
    /// just a predicate over the full pre-partition batch, which carries both columns as data -- no
    /// connector SQL engine and no partition overlay needed. A satisfying write commits and
    /// round-trips, with `region` reconstructed from `add.partitionValues`.
    #[tokio::test]
    async fn mixed_partition_and_data_constraint_validates_over_full_batch(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Two STRING columns so a partition-vs-data comparison is well-typed; `region` partitions.
        let schema = Arc::new(StructType::new_unchecked([
            StructField::nullable("label", DataType::STRING),
            StructField::nullable("region", DataType::STRING),
        ]));
        // Pre-partition batches carry both columns with their real per-row values.
        let mk_batch = |labels: Vec<&str>,
                        regions: Vec<&str>|
         -> Result<ArrowEngineData, Box<dyn std::error::Error>> {
            let arrow_schema = Arc::new(schema.as_ref().try_into_arrow()?);
            Ok(ArrowEngineData::new(RecordBatch::try_new(
                arrow_schema,
                vec![
                    Arc::new(StringArray::from(labels)) as ArrayRef,
                    Arc::new(StringArray::from(regions)) as ArrayRef,
                ],
            )?))
        };

        let (store, engine, table_location): (Arc<DynObjectStore>, _, _) =
            engine_store_setup("test_cc_mixed_partition", None);
        let table_url = create_table_with_configuration(
            store,
            table_location,
            schema.clone(),
            &["region"],
            true,
            vec![],
            vec!["checkConstraints"],
            HashMap::from([(
                "delta.constraints.region_ne_label".to_string(),
                "region != label".to_string(),
            )]),
        )
        .await?;

        let mut txn = begin_txn(&table_url, &engine)?.with_operation("WRITE".to_string());

        // The mixed constraint is kernel-parsable: a plain comparison over two batch columns.
        let constraints = txn.check_constraints();
        assert!(constraints.is_kernel_parsable());
        let constraint = constraints
            .iter()
            .exactly_one()
            .expect("table has exactly one constraint");
        assert!(constraint.predicate().is_some());

        let handler = engine.evaluation_handler();
        let validator = constraints.validator(handler.as_ref())?;

        // Satisfying: no row has region == label.
        validator.validate(&mk_batch(vec!["EU", "ASIA"], vec!["US", "US"])?)?;

        // Violating: row 1 has region == label ("US" == "US") -> `region != label` is false.
        let err = validator
            .validate(&mk_batch(vec!["EU", "US"], vec!["US", "US"])?)
            .expect_err("a row whose region equals its label must violate");
        match err {
            Error::CheckConstraintViolation {
                name,
                expression,
                details,
            } => {
                assert_eq!(name, "region_ne_label");
                assert_eq!(expression, "region != label");
                assert!(details.contains("row 1"), "locates the row: {details}");
            }
            other => panic!("expected CheckConstraintViolation, got: {other:?}"),
        }

        // A satisfying write commits and round-trips, with `region` reconstructed from
        // `add.partitionValues`.
        let good = mk_batch(vec!["EU", "ASIA"], vec!["US", "US"])?;
        validator.validate(&good)?;
        let write_context = txn.partitioned_write_context(HashMap::from([(
            "region".to_string(),
            Scalar::from("US"),
        )]))?;
        let add_files = engine.write_parquet(&good, &write_context).await?;
        txn.add_files(add_files);
        txn.commit(&engine)?.unwrap_committed();

        let expected = mk_batch(vec!["EU", "ASIA"], vec!["US", "US"])?;
        test_read(&expected, &table_url, Arc::new(engine))?;
        Ok(())
    }

    /// A `CheckConstraintValidator` binds evaluators once and validates many batches; a
    /// violation surfaces as the matchable `Error::CheckConstraintViolation` variant.
    #[tokio::test]
    async fn validator_binds_once_and_reports_typed_violations(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) = setup_constrained_table(
            "test_cc_validator_reuse",
            &[
                ("max_amount", "amount < 100"),
                ("positive_amount", "amount > 0"),
            ],
        )
        .await?;
        // No commit here -- this exercises the validator directly.
        let txn = begin_txn(&table_url, &engine)?;

        // Bind once...
        let evaluation_handler = engine.evaluation_handler();
        let validator = txn
            .check_constraints()
            .validator(evaluation_handler.as_ref())?;

        // ...validate many batches.
        validator.validate(&batch(vec![Some(1), Some(2)], vec!["a", "b"])?)?;
        validator.validate(&batch(vec![Some(50)], vec!["c"])?)?;

        let err = validator
            .validate(&batch(vec![Some(150)], vec!["d"])?)
            .expect_err("violating batch must error");
        match err {
            Error::CheckConstraintViolation {
                name,
                expression,
                details,
            } => {
                assert_eq!(name, "max_amount");
                assert_eq!(expression, "amount < 100");
                assert!(
                    details.contains("row 0"),
                    "details locate the row: {details}"
                );
                // Mirrors Delta-Spark: the violating row's referenced-column values appear.
                assert!(
                    details.contains("amount") && details.contains("150"),
                    "details include the violating values: {details}"
                );
            }
            other => panic!("expected CheckConstraintViolation, got: {other:?}"),
        }
        Ok(())
    }
}
