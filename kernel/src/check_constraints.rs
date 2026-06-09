//! Discovery and enforcement of CHECK constraints (in development, gated by the
//! `check-constraints-in-dev` cargo feature).
//!
//! CHECK constraints are boolean SQL expressions stored in the table configuration under
//! `delta.constraints.<name>`; the Delta protocol requires every row added to the table to
//! satisfy every constraint (a row passes only when a constraint evaluates to `true` -- both
//! `false` and `NULL` are violations). Kernel never sees row data on the write path, so
//! enforcement is a cooperative contract between kernel and the connector:
//!
//! - A connector acknowledges the contract by calling `Transaction::with_check_constraints`;
//!   without the acknowledgment, kernel fails writes to constrained tables so that connectors
//!   unaware of the feature cannot silently commit violating data.
//! - `Transaction::check_constraints` (and `WriteContext::check_constraints`) expose each
//!   constraint's raw SQL plus, when kernel can evaluate the expression, a kernel predicate.
//! - Each constraint reports where it is enforced via [`CheckConstraint::enforcement`]:
//!   - [`DataBatches`](CheckConstraintEnforcement::DataBatches): a [`CheckConstraintValidator`]
//!     binds the constraint to the engine's [`EvaluationHandler`] once per write and checks every
//!     batch, erroring with [`Error::CheckConstraintViolation`] on the first violating row. The
//!     default engine validates automatically in `write_parquet`; custom engines must validate (or
//!     enforce the raw SQL with their own evaluator) on every batch before writing.
//!   - [`PartitionValues`](CheckConstraintEnforcement::PartitionValues): kernel itself evaluates
//!     the constraint -- no engine needed -- against the partition values supplied to
//!     `Transaction::partitioned_write_context`, rejecting the write context on violation. Readers
//!     reconstruct partition columns from `add.partitionValues` rather than from file data, which
//!     makes the write context's partition values the protocol-correct enforcement point for such
//!     constraints.
//!   - [`Connector`](CheckConstraintEnforcement::Connector): the expression is outside the subset
//!     kernel's constraint parser supports (currently single column-vs-literal comparisons, e.g.
//!     `col1 < 10`); the connector must enforce the raw SQL itself, and kernel-driven validation
//!     fails closed.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
use crate::expressions::{parse_sql_simple_predicate, Scalar};
use crate::kernel_predicates::{DefaultKernelPredicateEvaluator, KernelPredicateEvaluator as _};
use crate::schema::{column_name, ColumnName, ColumnNamesAndTypes, DataType, SchemaRef};
use crate::utils::require;
use crate::{DeltaResult, EngineData, Error, EvaluationHandler, PredicateEvaluator, PredicateRef};

/// Table-configuration key prefix under which CHECK constraints are stored.
pub(crate) const CHECK_CONSTRAINT_PREFIX: &str = "delta.constraints.";

/// Returns true if the table configuration contains any CHECK constraints.
pub(crate) fn has_check_constraints(configuration: &HashMap<String, String>) -> bool {
    configuration
        .keys()
        .any(|key| key.starts_with(CHECK_CONSTRAINT_PREFIX))
}

/// Extracts all CHECK constraints from the table configuration, attempting to parse each one
/// against `schema`. Constraints kernel cannot evaluate are still returned (with
/// [`CheckConstraint::enforcement`] reporting [`CheckConstraintEnforcement::Connector`]) so that
/// connectors with their own SQL engine can enforce them from the raw SQL.
pub(crate) fn constraints_from_configuration(
    configuration: &HashMap<String, String>,
    schema: SchemaRef,
    partition_columns: &[String],
) -> Vec<CheckConstraint> {
    let mut constraints: Vec<_> = configuration
        .iter()
        .filter_map(|(key, sql)| {
            let name = key.strip_prefix(CHECK_CONSTRAINT_PREFIX)?;
            Some(CheckConstraint::new(
                name,
                sql,
                schema.clone(),
                partition_columns,
            ))
        })
        .collect();
    // HashMap iteration order is unstable; sort for deterministic discovery and error ordering.
    constraints.sort_by(|a, b| a.name.cmp(&b.name));
    constraints
}

/// Validates the (normalized, schema-cased) logical partition values of a write context against
/// every partition-value-enforced constraint. Kernel evaluates these directly -- partition
/// values are per-file constants, so no engine is needed. Constraints with other enforcement
/// kinds are skipped (data-batch constraints are validated per batch; connector-enforced
/// constraints fail closed when a validator is built).
pub(crate) fn enforce_on_partition_values(
    constraints: &[CheckConstraint],
    partition_values: &HashMap<String, Scalar>,
) -> DeltaResult<()> {
    let resolver: HashMap<ColumnName, Scalar> = partition_values
        .iter()
        .map(|(name, value)| (ColumnName::new([name]), value.clone()))
        .collect();
    let evaluator = DefaultKernelPredicateEvaluator::from(resolver);
    for constraint in constraints {
        let ConstraintSupport::PartitionValues { predicate, .. } = &constraint.support else {
            continue;
        };
        let result = evaluator.eval(predicate);
        if result != Some(true) {
            return Err(Error::CheckConstraintViolation {
                name: constraint.name.clone(),
                expression: constraint.raw_sql.clone(),
                details: format!(
                    "the write context's partition values evaluated to {}",
                    match result {
                        Some(_) => "false",
                        None => "NULL",
                    },
                ),
            });
        }
    }
    Ok(())
}

/// Where a CHECK constraint is enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckConstraintEnforcement {
    /// Kernel evaluates the constraint against every data batch, via a
    /// [`CheckConstraintValidator`] (the default engine does this automatically in
    /// `write_parquet`).
    DataBatches,
    /// The constraint references a partition column, so kernel evaluates it against the
    /// partition values of each partitioned write context (partition values are per-file
    /// constants, and readers reconstruct partition columns from `add.partitionValues`).
    /// Connectors that do not create kernel write contexts must enforce the constraint against
    /// their own partition values.
    PartitionValues,
    /// Kernel cannot evaluate the constraint; the connector must enforce the raw SQL with its
    /// own SQL engine or fail the write.
    Connector,
}

/// Whether (and where) kernel can evaluate a constraint's expression.
#[derive(Debug, Clone)]
enum ConstraintSupport {
    /// Kernel parsed the expression and evaluates it against data batches.
    DataBatches(PredicateRef),
    /// The expression references the named partition column; kernel evaluates it against the
    /// partition values of each partitioned write context.
    PartitionValues {
        predicate: PredicateRef,
        column: String,
    },
    /// The expression is outside the subset kernel's constraint parser supports.
    Unsupported,
}

/// One CHECK constraint: its name, the raw SQL stored under `delta.constraints.<name>`, and the
/// parsed kernel predicate when kernel can evaluate the expression.
#[derive(Debug, Clone)]
pub struct CheckConstraint {
    name: String,
    raw_sql: String,
    support: ConstraintSupport,
    // The logical schema the predicate was resolved against; batches validated against this
    // constraint must conform to it.
    schema: SchemaRef,
}

impl CheckConstraint {
    pub(crate) fn new(
        name: impl Into<String>,
        raw_sql: impl Into<String>,
        schema: SchemaRef,
        partition_columns: &[String],
    ) -> Self {
        let raw_sql = raw_sql.into();
        let support = match parse_sql_simple_predicate(&raw_sql, &schema) {
            Err(_) => ConstraintSupport::Unsupported,
            Ok(predicate) => {
                // The lowered predicate uses canonical (schema-cased) column names, and metadata
                // partition columns are schema-cased too; compare case-insensitively anyway to be
                // robust against non-canonical metadata.
                let partition_ref = predicate.references().into_iter().find_map(|column| {
                    let top_level = column.path().first()?;
                    partition_columns
                        .iter()
                        .find(|pc| pc.eq_ignore_ascii_case(top_level))
                        .cloned()
                });
                let predicate = Arc::new(predicate);
                match partition_ref {
                    Some(column) => ConstraintSupport::PartitionValues { predicate, column },
                    None => ConstraintSupport::DataBatches(predicate),
                }
            }
        };
        Self {
            name: name.into(),
            raw_sql,
            support,
            schema,
        }
    }

    /// The constraint's name (the suffix of its `delta.constraints.<name>` configuration key).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The constraint's boolean SQL expression, exactly as stored in the table configuration.
    pub fn raw_sql(&self) -> &str {
        &self.raw_sql
    }

    /// Where this constraint is enforced; see [`CheckConstraintEnforcement`].
    pub fn enforcement(&self) -> CheckConstraintEnforcement {
        match &self.support {
            ConstraintSupport::DataBatches(_) => CheckConstraintEnforcement::DataBatches,
            ConstraintSupport::PartitionValues { .. } => {
                CheckConstraintEnforcement::PartitionValues
            }
            ConstraintSupport::Unsupported => CheckConstraintEnforcement::Connector,
        }
    }

    /// The parsed kernel predicate, when kernel could parse the expression (the
    /// [`DataBatches`](CheckConstraintEnforcement::DataBatches) and
    /// [`PartitionValues`](CheckConstraintEnforcement::PartitionValues) enforcement kinds).
    pub fn predicate(&self) -> Option<&PredicateRef> {
        match &self.support {
            ConstraintSupport::DataBatches(predicate) => Some(predicate),
            ConstraintSupport::PartitionValues { predicate, .. } => Some(predicate),
            ConstraintSupport::Unsupported => None,
        }
    }

    /// Validates that every row of `batch` satisfies this constraint. Equivalent to building a
    /// single-constraint [`CheckConstraintValidator`]; prefer the validator to amortize evaluator
    /// construction when validating multiple batches or constraints.
    ///
    /// # Errors
    ///
    /// - The constraint is not batch-enforced: partition-value-enforced constraints are checked
    ///   when creating a partitioned write context, not against batches, and connector-enforced
    ///   constraints fail closed (see [`Self::enforcement`]).
    /// - [`Error::CheckConstraintViolation`] if any row evaluates to `false` or `NULL` (the
    ///   protocol counts both as violations).
    /// - The engine fails to evaluate the predicate.
    pub fn validate(
        &self,
        batch: &dyn EngineData,
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<()> {
        if let ConstraintSupport::PartitionValues { column, .. } = &self.support {
            return Err(Error::unsupported(format!(
                "CHECK constraint '{}' ({}) references partition column '{}' and is enforced \
                 against partition values when creating a partitioned write context, not against \
                 data batches",
                self.name, self.raw_sql, column
            )));
        }
        CheckConstraintValidator::try_new(std::slice::from_ref(self), evaluation_handler)?
            .validate(batch)
    }

    /// The fail-closed error for constraints kernel cannot evaluate at all.
    fn connector_enforced_error(&self) -> Error {
        Error::unsupported(format!(
            "CHECK constraint '{}' ({}) is outside the subset kernel can evaluate; the \
             connector must enforce it with its own SQL engine before writing",
            self.name, self.raw_sql
        ))
    }
}

/// A constraint's predicate evaluator, bound once via [`CheckConstraintValidator::try_new`].
struct BoundConstraint {
    name: String,
    raw_sql: String,
    evaluator: Arc<dyn PredicateEvaluator>,
}

/// The table's data-batch-enforced CHECK constraints bound to an engine's [`EvaluationHandler`],
/// ready to validate batches. Build it once per write and reuse it for every batch: construction
/// surfaces connector-enforced constraints immediately (fail closed, before any data is written)
/// and amortizes predicate-evaluator creation across batches.
///
/// Partition-value-enforced constraints are skipped here: kernel enforces them when creating
/// each partitioned write context. Connectors that do not create kernel write contexts must
/// enforce them against their own partition values.
///
/// Custom engines obtain one from `WriteContext::check_constraint_validator`, or directly from
/// the constraints returned by `Transaction::check_constraints`.
pub struct CheckConstraintValidator {
    bound: Vec<BoundConstraint>,
}

impl CheckConstraintValidator {
    /// Binds `constraints` to `evaluation_handler`, erroring (fail closed) if any constraint is
    /// connector-enforced. A connector that enforces such constraints with its own SQL engine
    /// should filter them out and bind only the remainder.
    pub fn try_new(
        constraints: &[CheckConstraint],
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<Self> {
        let bound = constraints
            .iter()
            .filter_map(|constraint| match &constraint.support {
                ConstraintSupport::DataBatches(predicate) => Some(
                    evaluation_handler
                        .new_predicate_evaluator(constraint.schema.clone(), predicate.clone())
                        .map(|evaluator| BoundConstraint {
                            name: constraint.name.clone(),
                            raw_sql: constraint.raw_sql.clone(),
                            evaluator,
                        }),
                ),
                // Enforced against the write context's partition values, not data batches.
                ConstraintSupport::PartitionValues { .. } => None,
                ConstraintSupport::Unsupported => Some(Err(constraint.connector_enforced_error())),
            })
            .collect::<DeltaResult<_>>()?;
        Ok(Self { bound })
    }

    /// Validates that every row of `batch` satisfies every bound constraint. `batch` must use
    /// the table's logical schema (the schema `WriteContext::logical_schema` reports, before any
    /// logical-to-physical transform), since constraints reference logical column names.
    ///
    /// # Errors
    ///
    /// [`Error::CheckConstraintViolation`] on the first row whose predicate does not evaluate to
    /// exactly `true` (`false` and `NULL` are both violations), or any engine evaluation error.
    pub fn validate(&self, batch: &dyn EngineData) -> DeltaResult<()> {
        self.bound.iter().try_for_each(|constraint| {
            let result = constraint.evaluator.evaluate(batch)?;
            let mut visitor = CheckResultVisitor {
                name: &constraint.name,
                raw_sql: &constraint.raw_sql,
                rows_visited: 0,
            };
            visitor.visit_rows_of(result.as_ref())
        })
    }
}

/// Visits the boolean "output" column produced by evaluating a constraint predicate, erroring on
/// the first row that is not exactly `true` (`NULL` counts as a violation, matching the
/// protocol's "must return true" rule and NOT NULL invariants).
struct CheckResultVisitor<'a> {
    name: &'a str,
    raw_sql: &'a str,
    rows_visited: usize,
}

impl RowVisitor for CheckResultVisitor<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> =
            LazyLock::new(|| (vec![column_name!("output")], vec![DataType::BOOLEAN]).into());
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        require!(
            getters.len() == 1,
            Error::InternalError(format!(
                "Wrong number of CheckResultVisitor getters: {}",
                getters.len()
            ))
        );
        for i in 0..row_count {
            let passed: Option<bool> = getters[0].get_opt(i, "check_constraint.output")?;
            if passed != Some(true) {
                return Err(Error::CheckConstraintViolation {
                    name: self.name.to_string(),
                    expression: self.raw_sql.to_string(),
                    details: format!(
                        "row {} of the batch evaluated to {}",
                        self.rows_visited + i,
                        match passed {
                            Some(_) => "false",
                            None => "NULL",
                        },
                    ),
                });
            }
        }
        self.rows_visited += row_count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{DataType, StructField, StructType};

    fn schema() -> SchemaRef {
        Arc::new(StructType::new_unchecked([
            StructField::nullable("amount", DataType::LONG),
            StructField::nullable("name", DataType::STRING),
        ]))
    }

    fn config(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn has_check_constraints_matches_prefix_only() {
        assert!(!has_check_constraints(&config(&[(
            "delta.appendOnly",
            "true"
        )])));
        assert!(has_check_constraints(&config(&[
            ("delta.appendOnly", "true"),
            ("delta.constraints.positive", "amount > 0"),
        ])));
    }

    #[test]
    fn discovery_extracts_sorted_constraints_and_skips_other_keys() {
        let config = config(&[
            ("delta.constraints.b_check", "amount < 100"),
            ("delta.constraints.a_check", "amount > 0"),
            ("delta.appendOnly", "true"),
        ]);
        let constraints = constraints_from_configuration(&config, schema(), &[]);
        let names: Vec<_> = constraints.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["a_check", "b_check"]);
        assert!(constraints
            .iter()
            .all(|c| c.enforcement() == CheckConstraintEnforcement::DataBatches));
        assert_eq!(constraints[0].raw_sql(), "amount > 0");
    }

    #[test]
    fn junctions_unknown_columns_and_functions_are_connector_enforced() {
        for sql in [
            "amount > 0 AND amount < 100", // junctions unsupported in the simple subset
            "nope > 0",                    // unknown column
            "length(name) > 0",            // function call
            "amount IS NOT NULL",          // null checks unsupported in the simple subset
        ] {
            let constraint = CheckConstraint::new("c", sql, schema(), &[]);
            assert_eq!(
                constraint.enforcement(),
                CheckConstraintEnforcement::Connector,
                "expected '{sql}' to be connector-enforced"
            );
            assert!(constraint.predicate().is_none());
            assert_eq!(constraint.raw_sql(), sql, "raw sql must round-trip");
        }
    }

    #[test]
    fn partition_column_constraints_are_partition_value_enforced() {
        let partition_columns = vec!["name".to_string()];
        // References the partition column (even with different casing).
        let on_partition = CheckConstraint::new("part", "NAME = 'a'", schema(), &partition_columns);
        assert_eq!(
            on_partition.enforcement(),
            CheckConstraintEnforcement::PartitionValues
        );
        assert!(on_partition.predicate().is_some());

        // A data-column constraint on the same partitioned table stays batch-enforced.
        let on_data = CheckConstraint::new("data", "amount > 0", schema(), &partition_columns);
        assert_eq!(
            on_data.enforcement(),
            CheckConstraintEnforcement::DataBatches
        );
    }

    #[test]
    fn enforce_on_partition_values_requires_true() {
        let partition_columns = vec!["name".to_string()];
        let constraints = constraints_from_configuration(
            &config(&[
                ("delta.constraints.name_check", "name = 'a'"),
                ("delta.constraints.positive_amount", "amount > 0"),
            ]),
            schema(),
            &partition_columns,
        );
        let values = |name: Scalar| HashMap::from([("name".to_string(), name)]);

        // Satisfying partition values pass; the data-batch constraint is skipped even though
        // `amount` is not a partition value.
        enforce_on_partition_values(&constraints, &values(Scalar::from("a"))).unwrap();

        // A false result is a violation.
        let err = enforce_on_partition_values(&constraints, &values(Scalar::from("b")))
            .expect_err("non-matching partition value must violate");
        let Error::CheckConstraintViolation { name, details, .. } = err else {
            panic!("expected CheckConstraintViolation, got: {err:?}");
        };
        assert_eq!(name, "name_check");
        assert!(details.contains("false"), "details report false: {details}");

        // A NULL partition value is also a violation (only `true` passes).
        let err =
            enforce_on_partition_values(&constraints, &values(Scalar::Null(DataType::STRING)))
                .expect_err("NULL partition value must violate");
        let Error::CheckConstraintViolation { details, .. } = err else {
            panic!("expected CheckConstraintViolation, got: {err:?}");
        };
        assert!(details.contains("NULL"), "details report NULL: {details}");
    }
}
