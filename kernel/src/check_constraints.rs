//! Discovery and enforcement of CHECK constraints (in development, gated by the
//! `check-constraints-in-dev` cargo feature).
//!
//! CHECK constraints are boolean SQL expressions stored in the table configuration under
//! `delta.constraints.<name>`; the Delta protocol requires every row added to the table to
//! satisfy every constraint (a row passes only when a constraint evaluates to `true` -- both
//! `false` and `NULL` are violations). Kernel never sees row data on the write path, so
//! enforcement is a cooperative contract between kernel and the connector:
//!
//! - A connector acknowledges the contract by calling `Transaction::check_constraints` -- the act
//!   of reading the table's constraints is the acknowledgment. Without it, kernel fails data-adding
//!   commits to constrained tables so that connectors unaware of the feature cannot silently commit
//!   violating data.
//! - `Transaction::check_constraints` (discovery + acknowledgment) and
//!   `Snapshot::check_constraints` (discovery only) expose each constraint's raw SQL plus, when
//!   kernel can evaluate the expression, a kernel predicate.
//! - Kernel parses each constraint it can ([`CheckConstraint::predicate`] is then `Some`) and
//!   enforces it when the connector runs a [`CheckConstraintValidator`] (built via
//!   [`CheckConstraints::validator`]) over its data, erroring with
//!   [`Error::CheckConstraintViolation`] on the first violating row. Each constraint is a plain
//!   predicate the engine's [`EvaluationHandler`] evaluates against a batch. The connector
//!   validates the full logical batch *before partitioning*, so partition columns are present as
//!   ordinary per-row data and need no special handling -- the value validated is the value that
//!   determines the partition, hence what readers reconstruct from `add.partitionValues`.
//!   Enforcement is not wired into `write_parquet` or any write-context step; the connector runs
//!   the validator itself, before it partitions or writes.
//! - A constraint kernel cannot parse ([`CheckConstraint::predicate`] is `None`, surfaced by
//!   [`CheckConstraints::connector_enforced`]) -- currently anything beyond a single
//!   column-vs-literal comparison like `col1 < 10` -- must be enforced by the connector's own SQL
//!   engine; kernel-driven validation fails closed.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
use crate::expressions::UnaryExpressionOp::ToJson;
use crate::expressions::{parse_sql_simple_predicate, Expression, Predicate};
use crate::schema::{
    column_name, ColumnName, ColumnNamesAndTypes, DataType, SchemaRef, StructField, StructType,
};
use crate::utils::require;
use crate::{
    DeltaResult, EngineData, Error, EvaluationHandler, ExpressionEvaluator, PredicateEvaluator,
    PredicateRef,
};

/// Table-configuration key prefix under which CHECK constraints are stored.
pub(crate) const CHECK_CONSTRAINT_PREFIX: &str = "delta.constraints.";

/// Returns the constraint name if `key` is a CHECK-constraint configuration key. Delta-Spark
/// matches the `delta.constraints.` prefix case-insensitively when discovering constraints, so
/// kernel must too -- otherwise kernel could ignore (and write past) a constraint other writers
/// enforce.
fn strip_constraint_prefix(key: &str) -> Option<&str> {
    let prefix = key.get(..CHECK_CONSTRAINT_PREFIX.len())?;
    prefix
        .eq_ignore_ascii_case(CHECK_CONSTRAINT_PREFIX)
        .then(|| &key[CHECK_CONSTRAINT_PREFIX.len()..])
}

/// Returns true if the table configuration contains any CHECK constraints.
pub(crate) fn has_check_constraints(configuration: &HashMap<String, String>) -> bool {
    configuration
        .keys()
        .any(|key| strip_constraint_prefix(key).is_some())
}

/// Extracts all CHECK constraints from the table configuration, attempting to parse each one
/// against `schema`. Constraints kernel cannot evaluate are still returned (with
/// [`CheckConstraint::predicate`] returning `None`) so that connectors with their own SQL engine
/// can enforce them from the raw SQL.
pub(crate) fn constraints_from_configuration(
    configuration: &HashMap<String, String>,
    schema: SchemaRef,
) -> CheckConstraints {
    let mut constraints: Vec<_> = configuration
        .iter()
        .filter_map(|(key, sql)| {
            let name = strip_constraint_prefix(key)?;
            Some(CheckConstraint::new(name, sql, schema.clone()))
        })
        .collect();
    // HashMap iteration order is unstable; sort for deterministic discovery and error ordering.
    constraints.sort_by(|a, b| a.name.cmp(&b.name));
    CheckConstraints(constraints.into())
}

/// All CHECK constraints on a table. Dereferences to a slice for per-constraint access.
///
/// The first question a connector asks is set-level, so it is answered here:
/// [`is_kernel_parsable`](Self::is_kernel_parsable) reports whether kernel parsed *every*
/// constraint. If it did, the connector builds a [`validator`](Self::validator) and runs it over
/// its data. If not, the connector must evaluate the remaining raw SQL itself --
/// [`connector_enforced`](Self::connector_enforced) yields exactly those constraints -- or fail
/// the write.
// `Arc<[_]>` (not `Vec`) so cloning the set is an O(1) refcount bump; callers (the transaction and
// snapshot discovery caches) hand out cheap clones of a single parse.
#[derive(Debug, Clone, Default)]
pub struct CheckConstraints(Arc<[CheckConstraint]>);

impl CheckConstraints {
    /// True if kernel parsed every constraint -- none require a connector to enforce them. The
    /// connector can then build a [`validator`](Self::validator) and let kernel evaluate all of
    /// them; otherwise it must handle the [`connector_enforced`](Self::connector_enforced)
    /// remainder itself.
    pub fn is_kernel_parsable(&self) -> bool {
        self.0.iter().all(|c| !c.is_connector_enforced())
    }

    /// The constraints kernel could not parse, whose [`raw_sql`](CheckConstraint::raw_sql) the
    /// connector must evaluate itself before writing (or fail the write). Empty when
    /// [`is_kernel_parsable`](Self::is_kernel_parsable) is true.
    pub fn connector_enforced(&self) -> impl Iterator<Item = &CheckConstraint> {
        self.0.iter().filter(|c| c.is_connector_enforced())
    }

    /// Binds every parsable constraint to `evaluation_handler` for batch validation (equivalent to
    /// [`CheckConstraintValidator::try_new`]). Fails closed if any constraint is
    /// connector-enforced; check [`is_kernel_parsable`](Self::is_kernel_parsable) first and
    /// handle the [`connector_enforced`](Self::connector_enforced) remainder to avoid that
    /// error.
    pub fn validator(
        &self,
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<CheckConstraintValidator> {
        CheckConstraintValidator::try_new(&self.0, evaluation_handler)
    }
}

impl std::ops::Deref for CheckConstraints {
    type Target = [CheckConstraint];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Whether kernel can evaluate a constraint's expression.
#[derive(Debug, Clone)]
enum ConstraintSupport {
    /// Kernel parsed the expression into a predicate it evaluates against each data batch. The
    /// batch carries every referenced column -- including partition columns, which the connector
    /// validates before partitioning, while they are still ordinary per-row data.
    Parsable(PredicateRef),
    /// The expression is outside the subset kernel's constraint parser supports; the payload is
    /// the parser's reason (e.g. an unresolved column vs. unsupported grammar).
    Unsupported(String),
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
    ) -> Self {
        let raw_sql = raw_sql.into();
        // Kernel evaluates a parsed constraint as a plain predicate over each data batch. Partition
        // columns need no special handling: the connector validates before partitioning, so they
        // are present in the batch as ordinary per-row data.
        let support = match parse_sql_simple_predicate(&raw_sql, &schema) {
            Ok(predicate) => ConstraintSupport::Parsable(Arc::new(predicate)),
            Err(reason) => ConstraintSupport::Unsupported(reason.to_string()),
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

    /// Whether kernel could not parse this constraint, so a connector must enforce its raw SQL.
    /// Equivalent to [`Self::predicate`] being `None`; centralizes the connector-enforced test the
    /// set-level [`CheckConstraints`] helpers use.
    fn is_connector_enforced(&self) -> bool {
        matches!(self.support, ConstraintSupport::Unsupported(_))
    }

    /// The parsed kernel predicate, or `None` when kernel could not parse the expression (the
    /// connector-enforced case). Present for every constraint kernel can enforce.
    pub fn predicate(&self) -> Option<&Predicate> {
        match &self.support {
            ConstraintSupport::Parsable(predicate) => Some(predicate),
            ConstraintSupport::Unsupported(_) => None,
        }
    }

    /// Validates that every row of `batch` satisfies this constraint. Equivalent to building a
    /// single-constraint [`CheckConstraintValidator`]; prefer the validator to amortize evaluator
    /// construction when validating multiple batches or constraints.
    ///
    /// # Errors
    ///
    /// - The constraint is connector-enforced (kernel could not parse it, so [`Self::predicate`] is
    ///   `None`): validation fails closed.
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

    /// The fail-closed error for a constraint kernel could not parse; the connector must enforce
    /// its raw SQL with its own engine. `reason` is the parser's explanation, preserved from
    /// [`ConstraintSupport::Unsupported`].
    fn connector_enforced_error(&self, reason: &str) -> Error {
        Error::unsupported(format!(
            "CHECK constraint '{}' ({}) cannot be evaluated by kernel ({}); the connector must \
             enforce it with its own SQL engine before writing",
            self.name, self.raw_sql, reason
        ))
    }
}

// Output column names of the two value-rendering stages; see [`ViolationValuesRenderer`].
const REFERENCED_COLUMN: &str = "referenced";
const VALUES_COLUMN: &str = "values";

/// Renders the columns a constraint references as one JSON object per row, used to include the
/// violating row's values in [`Error::CheckConstraintViolation`] (mirroring Delta-Spark's
/// violation messages). Rendering is best-effort: it runs only after a violation is found, and
/// any rendering failure simply omits the values from the error.
///
/// Two evaluator stages are needed because the JSON encoder requires a *named* struct column as
/// input: stage one materializes the referenced columns as a struct column (names come from the
/// output schema), stage two JSON-encodes that column to a single STRING column.
struct ViolationValuesRenderer {
    referenced: Arc<dyn ExpressionEvaluator>,
    to_json: Arc<dyn ExpressionEvaluator>,
}

impl ViolationValuesRenderer {
    fn try_new(
        schema: &SchemaRef,
        predicate: &Predicate,
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<Self> {
        // Sort the referenced columns for deterministic output (Delta-Spark does the same).
        let mut columns: Vec<&ColumnName> = predicate.references().into_iter().collect();
        columns.sort_by(|a, b| a.path().cmp(b.path()));

        let mut fields = Vec::with_capacity(columns.len());
        let mut field_exprs: Vec<Expression> = Vec::with_capacity(columns.len());
        for column in columns {
            let leaf = schema
                .walk_column_fields(column)?
                .last()
                .ok_or_else(|| Error::internal_error("empty column path in predicate"))?
                .data_type()
                .clone();
            // Flatten nested references to their dotted display name; the name only feeds the
            // rendered JSON keys.
            fields.push(StructField::nullable(column.path().join("."), leaf));
            field_exprs.push(column.clone().into());
        }
        let referenced_schema = Arc::new(StructType::new_unchecked([StructField::nullable(
            REFERENCED_COLUMN,
            DataType::Struct(Box::new(StructType::new_unchecked(fields))),
        )]));
        let referenced = evaluation_handler.new_expression_evaluator(
            schema.clone(),
            Arc::new(Expression::struct_from([Expression::struct_from(
                field_exprs,
            )])),
            DataType::Struct(Box::new(referenced_schema.as_ref().clone())),
        )?;

        let json_schema =
            StructType::new_unchecked([StructField::nullable(VALUES_COLUMN, DataType::STRING)]);
        let to_json = evaluation_handler.new_expression_evaluator(
            referenced_schema,
            Arc::new(Expression::struct_from([Expression::unary(
                ToJson,
                Expression::column([REFERENCED_COLUMN]),
            )])),
            DataType::Struct(Box::new(json_schema)),
        )?;

        Ok(Self {
            referenced,
            to_json,
        })
    }

    /// Renders the referenced-column values of `row` (an index into `batch`), or `None` if any
    /// step fails.
    fn render(&self, batch: &dyn EngineData, row: usize) -> Option<String> {
        let referenced = self.referenced.evaluate(batch).ok()?;
        let json = self.to_json.evaluate(referenced.as_ref()).ok()?;
        let mut visitor = StringAtRowVisitor {
            target_row: row,
            rows_visited: 0,
            value: None,
        };
        visitor.visit_rows_of(json.as_ref()).ok()?;
        visitor.value
    }
}

/// A constraint's predicate evaluator, bound once via [`CheckConstraintValidator::try_new`]. The
/// partition-value overlay (if any) is shared across constraints and lives on the validator, not
/// here.
struct BoundConstraint {
    name: String,
    raw_sql: String,
    evaluator: Arc<dyn PredicateEvaluator>,
    values_renderer: Option<ViolationValuesRenderer>,
}

/// The table's kernel-parsable CHECK constraints bound to an engine's [`EvaluationHandler`], ready
/// to validate batches. Build it once and reuse it for every batch: construction surfaces
/// connector-enforced constraints immediately (fail closed, before any data is written) and
/// amortizes predicate-evaluator creation across batches.
///
/// Every bound constraint is a plain predicate over a data batch. Because the connector validates
/// before partitioning, partition columns are present in the batch as ordinary per-row data, so
/// they need no special handling here.
///
/// Obtain one from [`CheckConstraints::validator`] (the set returned by
/// `Transaction::check_constraints` or `Snapshot::check_constraints`).
pub struct CheckConstraintValidator {
    bound: Vec<BoundConstraint>,
}

impl CheckConstraintValidator {
    /// Binds each parsable constraint's predicate to `evaluation_handler`, erroring (fail closed)
    /// on the first connector-enforced constraint. A connector that enforces those with its own
    /// SQL engine should filter them out (via [`CheckConstraints::connector_enforced`]) and bind
    /// only the remainder.
    pub fn try_new(
        constraints: &[CheckConstraint],
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<Self> {
        let mut bound = Vec::with_capacity(constraints.len());
        for constraint in constraints {
            let ConstraintSupport::Parsable(predicate) = &constraint.support else {
                let ConstraintSupport::Unsupported(reason) = &constraint.support else {
                    unreachable!("non-Parsable support is Unsupported");
                };
                return Err(constraint.connector_enforced_error(reason));
            };
            bound.push(BoundConstraint {
                name: constraint.name.clone(),
                raw_sql: constraint.raw_sql.clone(),
                evaluator: evaluation_handler
                    .new_predicate_evaluator(constraint.schema.clone(), predicate.clone())?,
                // Rendering violating values is a best-effort nicety; never fail binding over it.
                values_renderer: ViolationValuesRenderer::try_new(
                    &constraint.schema,
                    predicate,
                    evaluation_handler,
                )
                .ok(),
            });
        }
        Ok(Self { bound })
    }

    /// Validates that every row of `batch` satisfies every bound constraint. `batch` must use the
    /// table's logical schema, before any logical-to-physical transform, since constraints
    /// reference logical column names -- and it must still carry partition columns (the connector
    /// validates before partitioning drops them).
    ///
    /// # Errors
    ///
    /// [`Error::CheckConstraintViolation`] on the first row whose predicate does not evaluate to
    /// exactly `true` (`false` and `NULL` are both violations), or any engine evaluation error.
    pub fn validate(&self, batch: &dyn EngineData) -> DeltaResult<()> {
        for constraint in &self.bound {
            let result = constraint.evaluator.evaluate(batch)?;
            let mut visitor = CheckResultVisitor::default();
            visitor.visit_rows_of(result.as_ref())?;
            let Some(violation) = visitor.violation else {
                continue;
            };
            let values = constraint
                .values_renderer
                .as_ref()
                .and_then(|renderer| renderer.render(batch, violation.row))
                .map(|json| format!("; values: {json}"))
                .unwrap_or_default();
            return Err(Error::CheckConstraintViolation {
                name: constraint.name.clone(),
                expression: constraint.raw_sql.clone(),
                details: format!(
                    "row {} of the batch evaluated to {}{}",
                    violation.row,
                    if violation.result_was_null {
                        "NULL"
                    } else {
                        "false"
                    },
                    values,
                ),
            });
        }
        Ok(())
    }
}

/// The first violating row found while scanning a constraint's boolean "output" column.
struct FirstViolation {
    row: usize,
    result_was_null: bool,
}

/// Visits the boolean "output" column produced by evaluating a constraint predicate, recording
/// the first row that is not exactly `true` (`NULL` counts as a violation, matching the
/// protocol's "must return true" rule and NOT NULL invariants).
#[derive(Default)]
struct CheckResultVisitor {
    rows_visited: usize,
    violation: Option<FirstViolation>,
}

impl RowVisitor for CheckResultVisitor {
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
        if self.violation.is_none() {
            for i in 0..row_count {
                let passed: Option<bool> = getters[0].get_opt(i, "check_constraint.output")?;
                if passed != Some(true) {
                    self.violation = Some(FirstViolation {
                        row: self.rows_visited + i,
                        result_was_null: passed.is_none(),
                    });
                    break;
                }
            }
        }
        self.rows_visited += row_count;
        Ok(())
    }
}

/// Extracts the STRING "values" column at one target row.
struct StringAtRowVisitor {
    target_row: usize,
    rows_visited: usize,
    value: Option<String>,
}

impl RowVisitor for StringAtRowVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> =
            LazyLock::new(|| (vec![column_name!("values")], vec![DataType::STRING]).into());
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        require!(
            getters.len() == 1,
            Error::InternalError(format!(
                "Wrong number of StringAtRowVisitor getters: {}",
                getters.len()
            ))
        );
        if let Some(i) = self.target_row.checked_sub(self.rows_visited) {
            if i < row_count {
                self.value = getters[0].get_opt(i, "check_constraint.values")?;
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
    fn constraint_prefix_matches_case_insensitively() {
        // Delta-Spark discovers the prefix case-insensitively; kernel must not ignore a
        // constraint other writers enforce.
        assert!(!has_check_constraints(&config(&[(
            "delta.appendOnly",
            "true"
        )])));
        for key in [
            "delta.constraints.positive",
            "DELTA.CONSTRAINTS.positive",
            "Delta.Constraints.positive",
        ] {
            let config = config(&[(key, "amount > 0")]);
            assert!(has_check_constraints(&config), "prefix of {key} matches");
            let constraints = constraints_from_configuration(&config, schema());
            assert_eq!(constraints.len(), 1, "constraint under {key} discovered");
            assert_eq!(constraints[0].name(), "positive");
        }
    }

    #[test]
    fn collection_answers_the_set_level_parsable_question() {
        // All parsable (a data-column and a partition-column constraint alike -- both are just
        // predicates over the batch).
        let all_parsable = constraints_from_configuration(
            &config(&[
                ("delta.constraints.positive", "amount > 0"),
                ("delta.constraints.name_check", "name = 'a'"),
            ]),
            schema(),
        );
        assert!(all_parsable.is_kernel_parsable());
        assert_eq!(all_parsable.connector_enforced().count(), 0);

        // One constraint outside the supported grammar flips the set-level answer, and
        // connector_enforced() exposes exactly that constraint's raw SQL.
        let mixed = constraints_from_configuration(
            &config(&[
                ("delta.constraints.positive", "amount > 0"),
                ("delta.constraints.range", "amount > 0 AND amount < 100"),
            ]),
            schema(),
        );
        assert!(!mixed.is_kernel_parsable());
        let raw: Vec<_> = mixed
            .connector_enforced()
            .map(|c| (c.name(), c.raw_sql()))
            .collect();
        assert_eq!(raw, [("range", "amount > 0 AND amount < 100")]);

        // No constraints: trivially parsable.
        assert!(constraints_from_configuration(&config(&[]), schema()).is_kernel_parsable());
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
        assert!(constraints
            .iter()
            .all(|c| matches!(c.support, ConstraintSupport::Parsable(_))));
        assert_eq!(constraints[0].raw_sql(), "amount > 0");
    }

    #[test]
    fn token_spaced_and_parenthesized_expressions_are_evaluable() {
        // Delta-Spark stores parser-round-tripped, token-spaced expression text (e.g.
        // `concat ( num , text ) != '9i'`); simple comparisons must tolerate the same style.
        let constraint = CheckConstraint::new("p", "( amount > 0 )", schema());
        assert!(matches!(constraint.support, ConstraintSupport::Parsable(_)));
    }

    #[test]
    fn connector_enforced_errors_preserve_the_parser_reason() {
        // Unsupported grammar and unresolvable columns are different failure modes (Delta-Spark
        // raises distinct error classes); the fail-closed error must say which one applies.
        let error_for = |c: &CheckConstraint| {
            let ConstraintSupport::Unsupported(reason) = &c.support else {
                panic!("expected Unsupported");
            };
            c.connector_enforced_error(reason).to_string()
        };

        let junction = CheckConstraint::new("range", "amount > 0 AND amount < 100", schema());
        let msg = error_for(&junction);
        assert!(
            msg.contains("only simple comparison"),
            "junction reason surfaces: {msg}"
        );

        let unknown_column = CheckConstraint::new("ghost", "nope > 0", schema());
        let msg = error_for(&unknown_column);
        assert!(
            msg.contains("not found in schema"),
            "unresolved-column reason surfaces: {msg}"
        );
    }

    #[test]
    fn functions_and_null_checks_are_connector_enforced() {
        for sql in ["length(name) > 0", "amount IS NOT NULL"] {
            let constraint = CheckConstraint::new("c", sql, schema());
            assert!(
                matches!(constraint.support, ConstraintSupport::Unsupported(_)),
                "expected '{sql}' to be connector-enforced"
            );
            assert!(constraint.predicate().is_none());
            assert_eq!(constraint.raw_sql(), sql, "raw sql must round-trip");
        }
    }

    #[test]
    fn partition_and_mixed_column_constraints_parse_like_any_other() {
        // Partition membership no longer affects classification: the connector validates before
        // partitioning, so a partition column is ordinary batch data. A constraint on a partition
        // column (`name`), on a data column (`amount`), and one mixing both all parse.
        for sql in ["name = 'a'", "amount > 0", "name != amount"] {
            let c = CheckConstraint::new("c", sql, schema());
            assert!(
                matches!(c.support, ConstraintSupport::Parsable(_)),
                "'{sql}' should parse"
            );
            assert!(c.predicate().is_some());
        }

        // A junction is still outside the simple-comparison grammar -> connector-enforced.
        let junction = CheckConstraint::new("junction", "name = 'a' AND amount > 0", schema());
        assert!(matches!(
            junction.support,
            ConstraintSupport::Unsupported(_)
        ));
    }
}
