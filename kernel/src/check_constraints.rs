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
//!   constraint's raw SQL plus, when kernel can parse the expression, a kernel predicate.
//! - [`CheckConstraint::validate`] evaluates the parsed predicate against a batch via the engine's
//!   [`EvaluationHandler`] and errors on the first violating row. The default engine invokes this
//!   automatically in `write_parquet`; custom engines must call it (or enforce the raw SQL with
//!   their own evaluator) on every batch before writing.
//!
//! Kernel's constraint parser currently supports only single column-vs-literal comparisons
//! (e.g. `col1 < 10`). Constraints outside that subset are surfaced with
//! [`CheckConstraint::is_kernel_parsable`] returning false, and [`CheckConstraint::validate`]
//! fails closed for them.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
use crate::expressions::parse_sql_simple_predicate;
use crate::schema::{column_name, ColumnName, ColumnNamesAndTypes, DataType, SchemaRef};
use crate::utils::require;
use crate::{DeltaResult, EngineData, Error, EvaluationHandler, PredicateRef};

/// Table-configuration key prefix under which CHECK constraints are stored.
pub(crate) const CHECK_CONSTRAINT_PREFIX: &str = "delta.constraints.";

/// Returns true if the table configuration contains any CHECK constraints.
pub(crate) fn has_check_constraints(configuration: &HashMap<String, String>) -> bool {
    configuration
        .keys()
        .any(|key| key.starts_with(CHECK_CONSTRAINT_PREFIX))
}

/// Extracts all CHECK constraints from the table configuration, attempting to parse each one
/// against `schema`. Constraints kernel cannot parse are still returned (with
/// [`CheckConstraint::is_kernel_parsable`] false) so that connectors with their own SQL engine
/// can enforce them from the raw SQL.
pub(crate) fn constraints_from_configuration(
    configuration: &HashMap<String, String>,
    schema: SchemaRef,
) -> Vec<CheckConstraint> {
    let mut constraints: Vec<_> = configuration
        .iter()
        .filter_map(|(key, sql)| {
            let name = key.strip_prefix(CHECK_CONSTRAINT_PREFIX)?;
            Some(CheckConstraint::new(name, sql, schema.clone()))
        })
        .collect();
    // HashMap iteration order is unstable; sort for deterministic discovery and error ordering.
    constraints.sort_by(|a, b| a.name.cmp(&b.name));
    constraints
}

/// One CHECK constraint: its name, the raw SQL stored under `delta.constraints.<name>`, and the
/// parsed kernel predicate when kernel's parser supports the expression.
#[derive(Debug, Clone)]
pub struct CheckConstraint {
    name: String,
    raw_sql: String,
    // None when kernel cannot parse `raw_sql`; the connector must then enforce the raw SQL
    // itself (or fail the write).
    parsed: Option<PredicateRef>,
    // The logical schema the predicate was resolved against; batches passed to `validate` must
    // conform to it.
    schema: SchemaRef,
}

impl CheckConstraint {
    pub(crate) fn new(
        name: impl Into<String>,
        raw_sql: impl Into<String>,
        schema: SchemaRef,
    ) -> Self {
        let raw_sql = raw_sql.into();
        let parsed = parse_sql_simple_predicate(&raw_sql, &schema)
            .ok()
            .map(Arc::new);
        Self {
            name: name.into(),
            raw_sql,
            parsed,
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

    /// True if kernel parsed this constraint into a predicate, making [`Self::validate`] usable.
    /// When false, the connector must enforce [`Self::raw_sql`] with its own SQL engine or fail
    /// the write.
    pub fn is_kernel_parsable(&self) -> bool {
        self.parsed.is_some()
    }

    /// The parsed kernel predicate, when [`Self::is_kernel_parsable`] is true.
    pub fn predicate(&self) -> Option<&PredicateRef> {
        self.parsed.as_ref()
    }

    /// Validates that every row of `batch` satisfies this constraint by evaluating the parsed
    /// predicate with `evaluation_handler`. `batch` must use the table's logical schema (the
    /// schema `WriteContext::logical_schema` reports, before any logical-to-physical transform),
    /// since constraints reference logical column names.
    ///
    /// # Errors
    ///
    /// - The constraint is not kernel-parsable (fails closed; see [`Self::is_kernel_parsable`]).
    /// - Any row evaluates to `false` or `NULL` (the protocol counts both as violations).
    /// - The engine fails to evaluate the predicate.
    pub fn validate(
        &self,
        batch: &dyn EngineData,
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<()> {
        let Some(predicate) = &self.parsed else {
            return Err(Error::unsupported(format!(
                "CHECK constraint '{}' ({}) is not kernel-parsable; the connector must enforce \
                 it with its own SQL engine before writing",
                self.name, self.raw_sql
            )));
        };
        let evaluator =
            evaluation_handler.new_predicate_evaluator(self.schema.clone(), predicate.clone())?;
        let result = evaluator.evaluate(batch)?;
        let mut visitor = CheckResultVisitor {
            name: &self.name,
            raw_sql: &self.raw_sql,
            rows_visited: 0,
        };
        visitor.visit_rows_of(result.as_ref())
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
                return Err(Error::generic(format!(
                    "CHECK constraint '{}' ({}) violated by row {} of the batch (predicate \
                     evaluated to {})",
                    self.name,
                    self.raw_sql,
                    self.rows_visited + i,
                    match passed {
                        Some(_) => "false",
                        None => "NULL",
                    },
                )));
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
        let constraints = constraints_from_configuration(&config, schema());
        let names: Vec<_> = constraints.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["a_check", "b_check"]);
        assert!(constraints.iter().all(|c| c.is_kernel_parsable()));
        assert_eq!(constraints[0].raw_sql(), "amount > 0");
    }

    #[test]
    fn junctions_unknown_columns_and_functions_are_not_kernel_parsable() {
        for sql in [
            "amount > 0 AND amount < 100", // junctions unsupported in the simple subset
            "nope > 0",                    // unknown column
            "length(name) > 0",            // function call
            "amount IS NOT NULL",          // null checks unsupported in the simple subset
        ] {
            let constraint = CheckConstraint::new("c", sql, schema());
            assert!(
                !constraint.is_kernel_parsable(),
                "expected '{sql}' to be non-parsable"
            );
            assert_eq!(constraint.raw_sql(), sql, "raw sql must round-trip");
        }
    }
}
