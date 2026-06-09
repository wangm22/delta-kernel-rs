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
    use delta_kernel::arrow::array::{ArrayRef, Int64Array, StringArray};
    use delta_kernel::arrow::record_batch::RecordBatch;
    use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
    use delta_kernel::engine::arrow_data::ArrowEngineData;
    use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
    use delta_kernel::engine::default::DefaultEngine;
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
    async fn write_context_requires_acknowledgment() -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) =
            setup_constrained_table("test_cc_gate_wc", &[("positive_amount", "amount > 0")])
                .await?;
        // No with_check_constraints() => write-context creation fails closed.
        let txn = begin_txn(&table_url, &engine)?;
        let err = txn
            .unpartitioned_write_context()
            .expect_err("write context must require check-constraint acknowledgment");
        assert_err_contains(err, "with_check_constraints");
        Ok(())
    }

    #[tokio::test]
    async fn commit_requires_acknowledgment() -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) =
            setup_constrained_table("test_cc_gate_commit", &[("positive_amount", "amount > 0")])
                .await?;
        // Even a connector that skips kernel write contexts is stopped at commit.
        let txn = begin_txn(&table_url, &engine)?;
        let err = txn
            .commit(&engine)
            .expect_err("commit must require check-constraint acknowledgment");
        assert_err_contains(err, "with_check_constraints");
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
        let txn = begin_txn(&table_url, &engine)?.with_check_constraints();

        let constraints = txn.check_constraints();
        let constraint = constraints
            .iter()
            .exactly_one()
            .expect("table has exactly one constraint");
        assert_eq!(constraint.name(), "positive_amount");
        assert_eq!(constraint.raw_sql(), "amount > 0");
        assert!(constraint.is_kernel_evaluable());

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

    /// Approach B: a DefaultEngine connector gets per-batch enforcement automatically inside
    /// `write_parquet`; a satisfying write commits and round-trips.
    #[tokio::test]
    async fn default_engine_auto_enforces_on_write() -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) =
            setup_constrained_table("test_cc_auto_enforce", &[("positive_amount", "amount > 0")])
                .await?;
        let mut txn = begin_txn(&table_url, &engine)?
            .with_check_constraints()
            .with_operation("WRITE".to_string());
        let write_context = txn.unpartitioned_write_context()?;

        // Violating batch: rejected before any file is written.
        let bad = batch(vec![Some(1), Some(-5)], vec!["a", "b"])?;
        let err = engine
            .write_parquet(&bad, &write_context)
            .await
            .map(|_| ())
            .expect_err("write_parquet must reject a violating batch");
        assert_err_contains(err, "positive_amount");

        // Satisfying batch: written, committed, and readable.
        let good = batch(vec![Some(1), Some(5)], vec!["a", "b"])?;
        let add_files_metadata = engine.write_parquet(&good, &write_context).await?;
        txn.add_files(add_files_metadata);
        txn.commit(&engine)?.unwrap_committed();

        let expected = batch(vec![Some(1), Some(5)], vec!["a", "b"])?;
        test_read(&expected, &table_url, Arc::new(engine))?;
        Ok(())
    }

    /// Constraints kernel cannot parse (here: a junction, outside the simple-comparison
    /// subset) are surfaced via `is_kernel_evaluable` and fail the DefaultEngine path closed.
    #[tokio::test]
    async fn non_parsable_constraint_fails_closed() -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) = setup_constrained_table(
            "test_cc_non_parsable",
            &[("amount_range", "amount > 0 AND amount < 100")],
        )
        .await?;
        let txn = begin_txn(&table_url, &engine)?.with_check_constraints();

        // Discovery still exposes the raw SQL so strong connectors can self-enforce.
        let constraints = txn.check_constraints();
        let constraint = constraints
            .iter()
            .exactly_one()
            .expect("table has exactly one constraint");
        assert!(!constraint.is_kernel_evaluable());
        assert_eq!(constraint.raw_sql(), "amount > 0 AND amount < 100");

        // The DefaultEngine path cannot evaluate it, so it must not write at all.
        let write_context = txn.unpartitioned_write_context()?;
        let good = batch(vec![Some(1)], vec!["a"])?;
        let err = engine
            .write_parquet(&good, &write_context)
            .await
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

    /// Constraints referencing a partition column are not kernel-evaluable (partition values
    /// are per-file constants from the write context, not batch columns); constraints on data
    /// columns of the same partitioned table still are.
    #[tokio::test]
    async fn partition_column_constraint_not_evaluable() -> Result<(), Box<dyn std::error::Error>> {
        let (table_url, engine) = setup_constrained_table_partitioned(
            "test_cc_partition_col",
            &[
                ("name_check", "name = 'a'"),
                ("positive_amount", "amount > 0"),
            ],
            &["name"],
        )
        .await?;
        let txn = begin_txn(&table_url, &engine)?.with_check_constraints();
        let constraints = txn.check_constraints();

        let name_check = constraints
            .iter()
            .find(|c| c.name() == "name_check")
            .expect("name_check constraint exists");
        assert!(!name_check.is_kernel_evaluable());
        assert_eq!(name_check.raw_sql(), "name = 'a'"); // raw SQL exposed for self-enforcement

        let positive_amount = constraints
            .iter()
            .find(|c| c.name() == "positive_amount")
            .expect("positive_amount constraint exists");
        assert!(positive_amount.is_kernel_evaluable());

        // Validating the partition-column constraint fails closed with a targeted error.
        let data = batch(vec![Some(1)], vec!["a"])?;
        let err = name_check
            .validate(&data, engine.evaluation_handler().as_ref())
            .expect_err("partition-column constraint must fail closed");
        assert_err_contains(err, "partition column 'name'");
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
        let txn = begin_txn(&table_url, &engine)?.with_check_constraints();
        let write_context = txn.unpartitioned_write_context()?;

        // Bind once...
        let evaluation_handler = engine.evaluation_handler();
        let validator = write_context.check_constraint_validator(evaluation_handler.as_ref())?;

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
            }
            other => panic!("expected CheckConstraintViolation, got: {other:?}"),
        }
        Ok(())
    }
}
