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
//! - [`CheckConstraintValidator`] binds the evaluable constraints to the engine's
//!   [`EvaluationHandler`] once per write; its [`validate`](CheckConstraintValidator::validate)
//!   then checks each batch, erroring with [`Error::CheckConstraintViolation`] on the first
//!   violating row. The default engine validates automatically in `write_parquet`; custom engines
//!   must validate (or enforce the raw SQL with their own evaluator) on every batch before writing.
//!
//! A constraint is *kernel-evaluable* only if kernel's constraint parser supports its
//! expression (currently single column-vs-literal comparisons, e.g. `col1 < 10`) and it does
//! not reference a partition column (partition values are per-file constants supplied via the
//! write context, not columns of the data batch). Non-evaluable constraints are surfaced with
//! [`CheckConstraint::is_kernel_evaluable`] returning false and fail validation closed.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
use crate::expressions::parse_sql_simple_predicate;
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
/// [`CheckConstraint::is_kernel_evaluable`] false) so that connectors with their own SQL engine
/// can enforce them from the raw SQL.
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

/// Whether (and how) kernel can evaluate a constraint's expression.
#[derive(Debug, Clone)]
enum ConstraintSupport {
    /// Kernel parsed the expression and can evaluate it against data batches.
    Evaluable(PredicateRef),
    /// The expression is outside the subset kernel's constraint parser supports.
    UnsupportedExpression,
    /// The expression references the named partition column. Partition values are per-file
    /// constants supplied via the write context rather than columns of the data batch, so
    /// kernel does not evaluate such constraints against batches.
    ReferencesPartitionColumn(String),
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
            Err(_) => ConstraintSupport::UnsupportedExpression,
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
                match partition_ref {
                    Some(column) => ConstraintSupport::ReferencesPartitionColumn(column),
                    None => ConstraintSupport::Evaluable(Arc::new(predicate)),
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

    /// True if kernel can evaluate this constraint, making it usable with
    /// [`CheckConstraintValidator`]. When false -- the expression is outside the subset kernel's
    /// parser supports, or it references a partition column -- the connector must enforce
    /// [`Self::raw_sql`] with its own SQL engine or fail the write.
    pub fn is_kernel_evaluable(&self) -> bool {
        matches!(self.support, ConstraintSupport::Evaluable(_))
    }

    /// The parsed kernel predicate, when [`Self::is_kernel_evaluable`] is true.
    pub fn predicate(&self) -> Option<&PredicateRef> {
        match &self.support {
            ConstraintSupport::Evaluable(predicate) => Some(predicate),
            _ => None,
        }
    }

    /// Validates that every row of `batch` satisfies this constraint. Equivalent to building a
    /// single-constraint [`CheckConstraintValidator`]; prefer the validator to amortize evaluator
    /// construction when validating multiple batches or constraints.
    ///
    /// # Errors
    ///
    /// - The constraint is not kernel-evaluable (fails closed; see [`Self::is_kernel_evaluable`]).
    /// - [`Error::CheckConstraintViolation`] if any row evaluates to `false` or `NULL` (the
    ///   protocol counts both as violations).
    /// - The engine fails to evaluate the predicate.
    pub fn validate(
        &self,
        batch: &dyn EngineData,
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<()> {
        CheckConstraintValidator::try_new(std::slice::from_ref(self), evaluation_handler)?
            .validate(batch)
    }

    /// The fail-closed error explaining why kernel cannot evaluate this constraint.
    fn not_evaluable_error(&self) -> Error {
        match &self.support {
            ConstraintSupport::Evaluable(_) => Error::internal_error(format!(
                "CHECK constraint '{}' is evaluable; no error to report",
                self.name
            )),
            ConstraintSupport::UnsupportedExpression => Error::unsupported(format!(
                "CHECK constraint '{}' ({}) is outside the subset kernel can evaluate; the \
                 connector must enforce it with its own SQL engine before writing",
                self.name, self.raw_sql
            )),
            ConstraintSupport::ReferencesPartitionColumn(column) => Error::unsupported(format!(
                "CHECK constraint '{}' ({}) references partition column '{}'; kernel does not \
                 evaluate constraints over partition values, so the connector must enforce it \
                 before writing",
                self.name, self.raw_sql, column
            )),
        }
    }
}

/// A constraint's predicate evaluator, bound once via [`CheckConstraintValidator::try_new`].
struct BoundConstraint {
    name: String,
    raw_sql: String,
    evaluator: Arc<dyn PredicateEvaluator>,
}

/// The table's CHECK constraints bound to an engine's [`EvaluationHandler`], ready to validate
/// data batches. Build it once per write and reuse it for every batch: construction surfaces
/// non-evaluable constraints immediately (fail closed, before any data is written) and amortizes
/// predicate-evaluator creation across batches.
///
/// Custom engines obtain one from `WriteContext::check_constraint_validator`, or directly from
/// the constraints returned by `Transaction::check_constraints`.
pub struct CheckConstraintValidator {
    bound: Vec<BoundConstraint>,
}

impl CheckConstraintValidator {
    /// Binds `constraints` to `evaluation_handler`, erroring (fail closed) if any constraint is
    /// not kernel-evaluable. A connector that enforces non-evaluable constraints with its own
    /// SQL engine should filter them out and bind only the kernel-evaluable remainder.
    pub fn try_new(
        constraints: &[CheckConstraint],
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<Self> {
        let bound = constraints
            .iter()
            .map(|constraint| {
                let ConstraintSupport::Evaluable(predicate) = &constraint.support else {
                    return Err(constraint.not_evaluable_error());
                };
                Ok(BoundConstraint {
                    name: constraint.name.clone(),
                    raw_sql: constraint.raw_sql.clone(),
                    evaluator: evaluation_handler
                        .new_predicate_evaluator(constraint.schema.clone(), predicate.clone())?,
                })
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
        assert!(constraints.iter().all(|c| c.is_kernel_evaluable()));
        assert_eq!(constraints[0].raw_sql(), "amount > 0");
    }

    #[test]
    fn junctions_unknown_columns_and_functions_are_not_evaluable() {
        for sql in [
            "amount > 0 AND amount < 100", // junctions unsupported in the simple subset
            "nope > 0",                    // unknown column
            "length(name) > 0",            // function call
            "amount IS NOT NULL",          // null checks unsupported in the simple subset
        ] {
            let constraint = CheckConstraint::new("c", sql, schema(), &[]);
            assert!(
                !constraint.is_kernel_evaluable(),
                "expected '{sql}' to be non-evaluable"
            );
            assert_eq!(constraint.raw_sql(), sql, "raw sql must round-trip");
        }
    }

    #[test]
    fn partition_column_constraints_are_not_evaluable() {
        let partition_columns = vec!["name".to_string()];
        // References the partition column (even with different casing): not evaluable.
        let on_partition = CheckConstraint::new("part", "NAME = 'a'", schema(), &partition_columns);
        assert!(!on_partition.is_kernel_evaluable());
        assert!(on_partition.predicate().is_none());
        let err = on_partition.not_evaluable_error().to_string();
        assert!(
            err.contains("partition column 'name'"),
            "error must name the partition column: {err}"
        );

        // A data-column constraint on the same partitioned table stays evaluable.
        let on_data = CheckConstraint::new("data", "amount > 0", schema(), &partition_columns);
        assert!(on_data.is_kernel_evaluable());
    }
}
