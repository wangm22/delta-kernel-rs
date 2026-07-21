//! A/B benchmark for CHECK constraint enforcement, comparing two enforcement strategies.
//!
//! The binary is held constant (built with `check-constraints-in-dev`); we vary only the
//! enforcement strategy, the number of constraints, the batch size, and whether the batch
//! satisfies the constraints. The delta between strategies is the value of the optimization.
//!
//! # Strategies
//!
//! - `separate`: kernel's shipping model -- N constraints become N bound predicates, each evaluated
//!   against the batch and scanned for a violation independently (via
//!   [`CheckConstraints::validator`] / [`CheckConstraintValidator`]).
//! - `combined`: the optimization under test -- the N constraint predicates are folded into one
//!   `AND` predicate (`Predicate::and_from`), evaluated once and scanned once. Three-valued `AND`
//!   is `true` iff every conjunct is `true` (`false`/`NULL` are violations), so pass/fail semantics
//!   match `separate`. The tradeoff: `combined` cannot report *which* constraint failed, and (see
//!   below) cannot short-circuit across constraints on a violating batch.
//!
//! # Outcomes
//!
//! - `pass`: every row satisfies every constraint (full scan -- worst case for a passing write).
//! - `violate`: exactly one planted row violates exactly one constraint. Both the row position
//!   (never row 0) and which constraint it violates are chosen by a per-cell seeded RNG, so the
//!   sweep covers early/late failures. This is where the strategies can diverge: `separate` may
//!   short-circuit as soon as an early constraint's scan finds the bad row and skip the remaining
//!   constraints; `combined` always evaluates the whole `AND`. Seeding (not per-iteration
//!   randomness) is deliberate -- Criterion reuses one batch across a cell's iterations, so the
//!   input must be fixed within a cell for the timing to be meaningful.
//!
//! # Metrics
//!
//! - `validate_only`: build the validator + validate the batch. Isolates enforcement from I/O.
//! - `write_end_to_end`: acknowledge + validate + (on pass) write_parquet + commit. On `violate`
//!   the validate step errors, so no write/commit happens -- that is the realistic cost of a
//!   rejected write. Each timed iteration uses a fresh table (untimed setup) so a growing
//!   `_delta_log` cannot inflate later iterations.
//!
//! Constraints are `amount <> <sentinel>` comparisons (kernel-parsable), one distinct sentinel
//! each, so a row set to sentinel_k violates constraint k and nothing else.
//!
//! Run with:
//! ```bash
//! cargo bench -p delta_kernel --bench check_constraint_write_bench --features check-constraints-in-dev
//! ```

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "check-constraints-in-dev")]
mod enabled {
    use std::collections::HashMap;
    use std::sync::Arc;

    use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
    use delta_kernel::arrow::array::{Array as _, ArrayRef, BooleanArray, Int64Array, StringArray};
    use delta_kernel::arrow::record_batch::RecordBatch;
    use delta_kernel::committer::FileSystemCommitter;
    use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
    use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt as _};
    use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
    use delta_kernel::engine::default::DefaultEngine;
    use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
    use delta_kernel::{Engine as _, EngineData, Predicate, PredicateRef, Snapshot};
    use test_utils::{create_table_with_configuration, engine_store_setup};
    use tokio::runtime::Runtime;
    use url::Url;

    const ROW_COUNTS: &[usize] = &[10, 10_000, 100_000];
    const CONSTRAINT_COUNTS: &[usize] = &[1, 5, 20, 50, 100];

    // Base sentinel for `amount <> N` constraints. Chosen far outside the satisfying data range
    // (amounts are 1..=1000) so a passing batch never accidentally hits one.
    const SENTINEL_BASE: i64 = 10_000_000;

    #[derive(Clone, Copy)]
    enum Strategy {
        Separate,
        Combined,
        // Fast combined AND-check for the common passing case; on failure only, fall back to
        // per-constraint validation to recover which constraint was violated (the attribution the
        // pure `Combined` strategy loses). Passing writes pay just the combined cost.
        Hybrid,
    }
    impl Strategy {
        fn label(self) -> &'static str {
            match self {
                Strategy::Separate => "separate",
                Strategy::Combined => "combined",
                Strategy::Hybrid => "hybrid",
            }
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Outcome {
        Pass,
        Violate,
    }
    impl Outcome {
        fn label(self) -> &'static str {
            match self {
                Outcome::Pass => "pass",
                Outcome::Violate => "violate",
            }
        }
    }

    fn bench_schema() -> SchemaRef {
        Arc::new(
            StructType::try_new(vec![
                StructField::nullable("amount", DataType::LONG),
                StructField::nullable("name", DataType::STRING),
            ])
            .expect("valid schema"),
        )
    }

    // Deterministic per-cell RNG (splitmix64) so violation position/target are fixed within a cell
    // but vary across the sweep. Seeded from (rows, num_constraints) -- no external rand dep.
    fn seeded(rows: usize, n: usize) -> u64 {
        let mut x = 0x9E3779B97F4A7C15u64 ^ (rows as u64).wrapping_mul(0xD1B54A32D192ED03);
        x = x.wrapping_add((n as u64).wrapping_mul(0x94D049BB133111EB));
        x
    }
    fn next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    // `num_constraints` distinct `amount <> sentinel_i` constraints, keyed by name.
    fn constraint_config(num_constraints: usize) -> HashMap<String, String> {
        (0..num_constraints)
            .map(|i| {
                (
                    format!("delta.constraints.ne_{i}"),
                    format!("amount <> {}", SENTINEL_BASE + i as i64),
                )
            })
            .collect()
    }

    // A batch of `num_rows`. On `Violate`, one row (seeded position in [1, num_rows-1]) is set to
    // the sentinel of a seeded target constraint, so that row violates exactly that one constraint.
    fn make_batch(num_rows: usize, num_constraints: usize, outcome: Outcome) -> ArrowEngineData {
        let mut amounts: Vec<i64> = (0..num_rows as i64).map(|i| i % 1000 + 1).collect();
        if outcome == Outcome::Violate && num_rows > 1 && num_constraints > 0 {
            let mut rng = seeded(num_rows, num_constraints);
            let target = (next_u64(&mut rng) % num_constraints as u64) as i64;
            // Position in [1, num_rows-1]: never row 0, so short-circuit scanning is exercised.
            let pos = 1 + (next_u64(&mut rng) % (num_rows as u64 - 1)) as usize;
            amounts[pos] = SENTINEL_BASE + target;
        }
        let names: Vec<String> = (0..num_rows).map(|i| format!("row_{i}")).collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(amounts)),
            Arc::new(StringArray::from(names)),
        ];
        let arrow_schema = Arc::new(
            bench_schema()
                .as_ref()
                .try_into_arrow()
                .expect("arrow schema"),
        );
        ArrowEngineData::new(RecordBatch::try_new(arrow_schema, columns).expect("valid batch"))
    }

    async fn setup_table(
        test_name: &str,
        num_constraints: usize,
    ) -> (Url, DefaultEngine<TokioBackgroundExecutor>) {
        let (store, engine, table_location) = engine_store_setup(test_name, None);
        let (writer_features, config) = if num_constraints == 0 {
            (vec![], HashMap::new())
        } else {
            (vec!["checkConstraints"], constraint_config(num_constraints))
        };
        let table_url = create_table_with_configuration(
            store,
            table_location,
            bench_schema(),
            &[],
            true,
            vec![],
            writer_features,
            config,
        )
        .await
        .expect("create table");
        (table_url, engine)
    }

    // Fold every constraint predicate into one AND. Errors if any constraint is connector-enforced
    // (none are here -- all `amount <> N` parse), matching `validator()`'s fail-closed contract.
    fn combined_predicate(
        constraints: &delta_kernel::check_constraints::CheckConstraints,
    ) -> PredicateRef {
        let preds: Vec<Predicate> = constraints
            .iter()
            .map(|c| {
                c.predicate()
                    .expect("all bench constraints are kernel-parsable")
                    .clone()
            })
            .collect();
        Arc::new(Predicate::and_from(preds))
    }

    // The `combined` strategy's validate: evaluate the single AND predicate, then scan its boolean
    // output for any row that is not exactly `true`. Mirrors what a combined validator would do;
    // the linear scan is O(rows), same order as kernel's per-constraint visitor scan.
    fn combined_validate(
        evaluator: &dyn delta_kernel::PredicateEvaluator,
        batch: &dyn EngineData,
    ) -> bool {
        let out = evaluator.evaluate(batch).expect("evaluate");
        let rb = out.try_into_record_batch().expect("bool batch");
        let col = rb
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean output");
        // Passes iff every row is exactly `true`: no nulls and every valid value is true. Any
        // null (unknown) or false is a violation, matching kernel's three-valued semantics.
        col.null_count() == 0 && col.true_count() == col.len()
    }

    fn build_combined_evaluator(
        engine: &DefaultEngine<TokioBackgroundExecutor>,
        constraints: &delta_kernel::check_constraints::CheckConstraints,
    ) -> Arc<dyn delta_kernel::PredicateEvaluator> {
        let handler = engine.evaluation_handler();
        let predicate = combined_predicate(constraints);
        handler
            .new_predicate_evaluator(bench_schema(), predicate)
            .expect("combined evaluator")
    }

    // The `hybrid` strategy: run the fast single-AND check first. If it passes (the common case),
    // done -- same cost as `combined`. Only on failure fall back to the per-constraint `validator`,
    // which pinpoints and names the violated constraint (the attribution `combined` cannot give).
    // Returns Ok(()) on pass, or the attributed violation error on failure.
    fn hybrid_validate(
        combined: &dyn delta_kernel::PredicateEvaluator,
        constraints: &delta_kernel::check_constraints::CheckConstraints,
        handler: &dyn delta_kernel::EvaluationHandler,
        batch: &dyn EngineData,
    ) -> delta_kernel::DeltaResult<()> {
        if combined_validate(combined, batch) {
            return Ok(());
        }
        // Fast path said "some constraint failed"; recover which one via the per-constraint path.
        constraints.validator(handler)?.validate(batch)
    }

    pub fn run(c: &mut Criterion) {
        let rt = Runtime::new().expect("tokio runtime");

        // ===================== validate_only: isolate enforcement =====================
        let mut group = c.benchmark_group("validate_only");
        for &rows in ROW_COUNTS {
            group.throughput(Throughput::Elements(rows as u64));
            for &n in CONSTRAINT_COUNTS {
                let (url, engine) = rt.block_on(setup_table(&format!("vo_{n}c_{rows}"), n));
                let snapshot = Snapshot::builder_for(url.clone())
                    .build(&engine)
                    .expect("snapshot");
                let handler = engine.evaluation_handler();

                for outcome in [Outcome::Pass, Outcome::Violate] {
                    let data = make_batch(rows, n, outcome);
                    for strategy in [Strategy::Separate, Strategy::Combined, Strategy::Hybrid] {
                        let id = BenchmarkId::new(
                            format!("{}/{}/{}c", strategy.label(), outcome.label(), n),
                            rows,
                        );
                        match strategy {
                            Strategy::Separate => group.bench_with_input(id, &rows, |b, _| {
                                b.iter(|| {
                                    let txn = snapshot
                                        .clone()
                                        .transaction(Box::new(FileSystemCommitter::new()), &engine)
                                        .expect("txn");
                                    let constraints = txn.check_constraints();
                                    let validator =
                                        constraints.validator(handler.as_ref()).expect("validator");
                                    // Ok(pass) or Err(violation); both are valid measured outcomes.
                                    let _ = validator.validate(&data as &dyn EngineData);
                                });
                            }),
                            Strategy::Combined => group.bench_with_input(id, &rows, |b, _| {
                                b.iter(|| {
                                    let txn = snapshot
                                        .clone()
                                        .transaction(Box::new(FileSystemCommitter::new()), &engine)
                                        .expect("txn");
                                    let constraints = txn.check_constraints();
                                    let evaluator = build_combined_evaluator(&engine, &constraints);
                                    let _ = combined_validate(evaluator.as_ref(), &data);
                                });
                            }),
                            Strategy::Hybrid => group.bench_with_input(id, &rows, |b, _| {
                                b.iter(|| {
                                    let txn = snapshot
                                        .clone()
                                        .transaction(Box::new(FileSystemCommitter::new()), &engine)
                                        .expect("txn");
                                    let constraints = txn.check_constraints();
                                    let evaluator = build_combined_evaluator(&engine, &constraints);
                                    // Ok(pass) or Err(attributed violation); both valid outcomes.
                                    let _ = hybrid_validate(
                                        evaluator.as_ref(),
                                        &constraints,
                                        handler.as_ref(),
                                        &data as &dyn EngineData,
                                    );
                                });
                            }),
                        };
                    }
                }
            }
        }
        group.finish();

        // ===================== write_end_to_end: full write path =====================
        // Fresh table per timed iteration (untimed setup) so log growth cannot inflate timings.
        let mut group = c.benchmark_group("write_end_to_end");
        for &rows in ROW_COUNTS {
            group.throughput(Throughput::Elements(rows as u64));

            // Unconstrained reference (no constraints, no validation).
            let pass_data = make_batch(rows, 0, Outcome::Pass);
            group.bench_with_input(
                BenchmarkId::new("unconstrained/pass/0c", rows),
                &rows,
                |b, _| {
                    b.iter_batched(
                        || rt.block_on(setup_table("ee_none", 0)),
                        |(url, engine)| {
                            let snapshot =
                                Snapshot::builder_for(url).build(&engine).expect("snapshot");
                            let mut txn = snapshot
                                .transaction(Box::new(FileSystemCommitter::new()), &engine)
                                .expect("txn")
                                .with_operation("INSERT".to_string());
                            let wc = txn.unpartitioned_write_context().expect("wc");
                            let add = rt
                                .block_on(engine.write_parquet(&pass_data, &wc))
                                .expect("write");
                            txn.add_files(add);
                            txn.commit(&engine).expect("commit").unwrap_committed();
                        },
                        BatchSize::PerIteration,
                    );
                },
            );

            for &n in CONSTRAINT_COUNTS {
                for outcome in [Outcome::Pass, Outcome::Violate] {
                    let data = make_batch(rows, n, outcome);
                    for strategy in [Strategy::Separate, Strategy::Combined, Strategy::Hybrid] {
                        let id = BenchmarkId::new(
                            format!("{}/{}/{}c", strategy.label(), outcome.label(), n),
                            rows,
                        );
                        group.bench_with_input(id, &rows, |b, _| {
                            b.iter_batched(
                                || rt.block_on(setup_table("ee_c", n)),
                                |(url, engine)| {
                                    run_e2e(&rt, &engine, url, &data, strategy);
                                },
                                BatchSize::PerIteration,
                            );
                        });
                    }
                }
            }
        }
        group.finish();
    }

    // One end-to-end constrained write. On a passing batch: validate, write, commit. On a violating
    // batch: validate errors and we stop (a rejected write does no I/O) -- the realistic
    // failed-write cost. Returns nothing; panics only on unexpected (non-violation) errors.
    fn run_e2e(
        rt: &Runtime,
        engine: &DefaultEngine<TokioBackgroundExecutor>,
        url: Url,
        data: &ArrowEngineData,
        strategy: Strategy,
    ) {
        let snapshot = Snapshot::builder_for(url).build(engine).expect("snapshot");
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), engine)
            .expect("txn")
            .with_operation("INSERT".to_string());
        let handler = engine.evaluation_handler();
        let constraints = txn.check_constraints();

        let passed = match strategy {
            Strategy::Separate => {
                let validator = constraints.validator(handler.as_ref()).expect("validator");
                validator.validate(data as &dyn EngineData).is_ok()
            }
            Strategy::Combined => {
                let evaluator = build_combined_evaluator(engine, &constraints);
                combined_validate(evaluator.as_ref(), data)
            }
            Strategy::Hybrid => {
                let evaluator = build_combined_evaluator(engine, &constraints);
                hybrid_validate(evaluator.as_ref(), &constraints, handler.as_ref(), data).is_ok()
            }
        };
        if !passed {
            return; // rejected write: no parquet, no commit
        }
        let wc = txn.unpartitioned_write_context().expect("wc");
        let add = rt.block_on(engine.write_parquet(data, &wc)).expect("write");
        txn.add_files(add);
        txn.commit(engine).expect("commit").unwrap_committed();
    }
}

#[cfg(feature = "check-constraints-in-dev")]
fn benches(c: &mut Criterion) {
    enabled::run(c);
}

// Without the feature, writes to constrained tables are unsupported; nothing to measure.
#[cfg(not(feature = "check-constraints-in-dev"))]
fn benches(_c: &mut Criterion) {}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
