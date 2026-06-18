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
use crate::expressions::UnaryExpressionOp::ToJson;
use crate::expressions::{parse_sql_simple_predicate, Expression, Predicate, Scalar};
use crate::kernel_predicates::{DefaultKernelPredicateEvaluator, KernelPredicateEvaluator as _};
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
/// [`CheckConstraint::enforcement`] reporting [`CheckConstraintEnforcement::Connector`]) so that
/// connectors with their own SQL engine can enforce them from the raw SQL.
pub(crate) fn constraints_from_configuration(
    configuration: &HashMap<String, String>,
    schema: SchemaRef,
    partition_columns: &[String],
) -> CheckConstraints {
    let mut constraints: Vec<_> = configuration
        .iter()
        .filter_map(|(key, sql)| {
            let name = strip_constraint_prefix(key)?;
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
    CheckConstraints(constraints)
}

/// All CHECK constraints on a table. Dereferences to a slice for per-constraint access.
///
/// The first question a connector asks is set-level, so it is answered here:
/// [`is_kernel_parsable`](Self::is_kernel_parsable) reports whether kernel parsed *every*
/// constraint. If it did, kernel enforces them all (per data batch or per partitioned write
/// context) and the connector proceeds normally. If not, the connector must evaluate the
/// remaining raw SQL itself -- [`connector_enforced`](Self::connector_enforced) yields exactly
/// those constraints -- or fail the write.
#[derive(Debug, Clone, Default)]
pub struct CheckConstraints(Vec<CheckConstraint>);

impl CheckConstraints {
    /// True if kernel parsed every constraint, i.e. no constraint requires
    /// [`Connector`](CheckConstraintEnforcement::Connector) enforcement. Kernel then enforces
    /// all of them: data-batch constraints via [`CheckConstraintValidator`] (automatic in the
    /// default engine's `write_parquet`) and partition-column constraints when each partitioned
    /// write context is created.
    pub fn is_kernel_parsable(&self) -> bool {
        self.0
            .iter()
            .all(|c| c.enforcement() != CheckConstraintEnforcement::Connector)
    }

    /// The constraints kernel could not parse, whose [`raw_sql`](CheckConstraint::raw_sql) the
    /// connector must evaluate itself before writing (or fail the write). Empty when
    /// [`is_kernel_parsable`](Self::is_kernel_parsable) is true.
    pub fn connector_enforced(&self) -> impl Iterator<Item = &CheckConstraint> {
        self.0
            .iter()
            .filter(|c| c.enforcement() == CheckConstraintEnforcement::Connector)
    }

    /// Binds the data-batch-enforced constraints to `evaluation_handler` for per-batch
    /// validation; equivalent to [`CheckConstraintValidator::try_new`]. Fails closed if any
    /// constraint is connector-enforced -- check [`is_kernel_parsable`](Self::is_kernel_parsable)
    /// first and handle the [`connector_enforced`](Self::connector_enforced) remainder to avoid
    /// that error.
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
            // Report the referenced partition values alongside the verdict (mirrors
            // Delta-Spark's violation messages, which list the violating values).
            let mut referenced: Vec<String> = predicate
                .references()
                .into_iter()
                .filter_map(|column| {
                    let top_level = column.path().first()?;
                    let value = partition_values.get(top_level)?;
                    Some(format!("{top_level} = {value}"))
                })
                .collect();
            referenced.sort();
            return Err(Error::CheckConstraintViolation {
                name: constraint.name.clone(),
                expression: constraint.raw_sql.clone(),
                details: format!(
                    "the write context's partition values ({}) evaluated to {}",
                    referenced.join(", "),
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
        partition_columns: &[String],
    ) -> Self {
        let raw_sql = raw_sql.into();
        let support = match parse_sql_simple_predicate(&raw_sql, &schema) {
            Err(reason) => ConstraintSupport::Unsupported(reason.to_string()),
            Ok(predicate) => {
                // Classify by which columns the predicate references. The lowered predicate uses
                // canonical (schema-cased) column names, and metadata partition columns are
                // schema-cased too; compare case-insensitively anyway to be robust against
                // non-canonical metadata.
                let references = predicate.references();
                let is_partition = |column: &ColumnName| {
                    column.path().first().is_some_and(|top_level| {
                        partition_columns
                            .iter()
                            .any(|pc| pc.eq_ignore_ascii_case(top_level))
                    })
                };
                let partition_column = references.iter().find(|c| is_partition(c)).map(|column| {
                    // Render with the metadata-cased partition column name.
                    let top_level = &column.path()[0];
                    partition_columns
                        .iter()
                        .find(|pc| pc.eq_ignore_ascii_case(top_level))
                        .cloned()
                        .unwrap_or_else(|| top_level.clone())
                });
                let references_data_column = references.iter().any(|c| !is_partition(c));
                let predicate = Arc::new(predicate);
                match partition_column {
                    // A single constraint that references both a partition column and a data
                    // column can be evaluated against partition values OR data batches, but not a
                    // mix -- neither enforcement path sees all the columns. Rather than enforce it
                    // half-correctly, mark it connector-enforced so kernel-evaluating engines fail
                    // closed (engines with their own evaluator can still enforce the raw SQL).
                    Some(_) if references_data_column => ConstraintSupport::Unsupported(
                        "constraint references both partition and data columns, which kernel \
                         cannot evaluate against a single source"
                            .to_string(),
                    ),
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
            ConstraintSupport::Unsupported(_) => CheckConstraintEnforcement::Connector,
        }
    }

    /// The parsed kernel predicate, when kernel could parse the expression (the
    /// [`DataBatches`](CheckConstraintEnforcement::DataBatches) and
    /// [`PartitionValues`](CheckConstraintEnforcement::PartitionValues) enforcement kinds).
    pub fn predicate(&self) -> Option<&PredicateRef> {
        match &self.support {
            ConstraintSupport::DataBatches(predicate) => Some(predicate),
            ConstraintSupport::PartitionValues { predicate, .. } => Some(predicate),
            ConstraintSupport::Unsupported(_) => None,
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
        let reason = match &self.support {
            ConstraintSupport::Unsupported(reason) => reason.as_str(),
            _ => "constraint is kernel-evaluable",
        };
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

/// A constraint's predicate evaluator, bound once via [`CheckConstraintValidator::try_new`].
struct BoundConstraint {
    name: String,
    raw_sql: String,
    evaluator: Arc<dyn PredicateEvaluator>,
    values_renderer: Option<ViolationValuesRenderer>,
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
                            // Rendering violating values is a best-effort nicety; never fail
                            // binding over it.
                            values_renderer: ViolationValuesRenderer::try_new(
                                &constraint.schema,
                                predicate,
                                evaluation_handler,
                            )
                            .ok(),
                        }),
                ),
                // Enforced against the write context's partition values, not data batches.
                ConstraintSupport::PartitionValues { .. } => None,
                ConstraintSupport::Unsupported(_) => {
                    Some(Err(constraint.connector_enforced_error()))
                }
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
            let constraints = constraints_from_configuration(&config, schema(), &[]);
            assert_eq!(constraints.len(), 1, "constraint under {key} discovered");
            assert_eq!(constraints[0].name(), "positive");
        }
    }

    #[test]
    fn collection_answers_the_set_level_parsable_question() {
        // All parsable (including a partition-column constraint, which kernel also enforces).
        let all_parsable = constraints_from_configuration(
            &config(&[
                ("delta.constraints.positive", "amount > 0"),
                ("delta.constraints.name_check", "name = 'a'"),
            ]),
            schema(),
            &["name".to_string()],
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
            &[],
        );
        assert!(!mixed.is_kernel_parsable());
        let raw: Vec<_> = mixed
            .connector_enforced()
            .map(|c| (c.name(), c.raw_sql()))
            .collect();
        assert_eq!(raw, [("range", "amount > 0 AND amount < 100")]);

        // No constraints: trivially parsable.
        assert!(constraints_from_configuration(&config(&[]), schema(), &[]).is_kernel_parsable());
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
    fn token_spaced_and_parenthesized_expressions_are_evaluable() {
        // Delta-Spark stores parser-round-tripped, token-spaced expression text (e.g.
        // `concat ( num , text ) != '9i'`); simple comparisons must tolerate the same style.
        let constraint = CheckConstraint::new("p", "( amount > 0 )", schema(), &[]);
        assert_eq!(
            constraint.enforcement(),
            CheckConstraintEnforcement::DataBatches
        );
    }

    #[test]
    fn connector_enforced_errors_preserve_the_parser_reason() {
        // Unsupported grammar and unresolvable columns are different failure modes (Delta-Spark
        // raises distinct error classes); the fail-closed error must say which one applies.
        let junction = CheckConstraint::new("range", "amount > 0 AND amount < 100", schema(), &[]);
        assert_eq!(
            junction.enforcement(),
            CheckConstraintEnforcement::Connector
        );
        let msg = junction.connector_enforced_error().to_string();
        assert!(
            msg.contains("only simple comparison"),
            "junction reason surfaces: {msg}"
        );

        let unknown_column = CheckConstraint::new("ghost", "nope > 0", schema(), &[]);
        let msg = unknown_column.connector_enforced_error().to_string();
        assert!(
            msg.contains("not found in schema"),
            "unresolved-column reason surfaces: {msg}"
        );
    }

    #[test]
    fn functions_and_null_checks_are_connector_enforced() {
        for sql in ["length(name) > 0", "amount IS NOT NULL"] {
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
    fn constraint_mixing_partition_and_data_columns_is_connector_enforced() {
        let partition_columns = vec!["name".to_string()];
        // References both the partition column (`name`) and a data column (`amount`). Neither the
        // partition-values path nor the data-batch path sees all columns, so kernel cannot
        // evaluate it against a single source -- it is reported connector-enforced (fail closed).
        let mixed = CheckConstraint::new(
            "mixed",
            "name = 'a' AND amount > 0",
            schema(),
            &partition_columns,
        );
        // Connector-enforced: kernel cannot evaluate it against a single source.
        assert_eq!(mixed.enforcement(), CheckConstraintEnforcement::Connector);
        assert!(mixed.predicate().is_none());
        // The raw SQL stays exposed so a strong connector can still enforce it.
        assert_eq!(mixed.raw_sql(), "name = 'a' AND amount > 0");
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

        // A false result is a violation, and the details report the offending values.
        let err = enforce_on_partition_values(&constraints, &values(Scalar::from("b")))
            .expect_err("non-matching partition value must violate");
        let Error::CheckConstraintViolation { name, details, .. } = err else {
            panic!("expected CheckConstraintViolation, got: {err:?}");
        };
        assert_eq!(name, "name_check");
        assert!(details.contains("false"), "details report false: {details}");
        assert!(
            details.contains("name = "),
            "details include the partition value: {details}"
        );

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
