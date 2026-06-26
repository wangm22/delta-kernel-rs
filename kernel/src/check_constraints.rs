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
//!     the constraint -- no engine needed -- against the write context's partition values, once,
//!     when a validator is built from a partitioned write context (the default engine does this in
//!     `write_parquet`). Readers reconstruct partition columns from `add.partitionValues` rather
//!     than from file data, which makes the write context's partition values the protocol-correct
//!     enforcement point for such constraints.
//!   - [`Connector`](CheckConstraintEnforcement::Connector): the expression is outside the subset
//!     kernel's constraint parser supports (currently single column-vs-literal comparisons, e.g.
//!     `col1 < 10`); the connector must enforce the raw SQL itself, and kernel-driven validation
//!     fails closed.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock};

use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
use crate::expressions::UnaryExpressionOp::ToJson;
use crate::expressions::{parse_sql_simple_predicate, Expression, Predicate, Scalar};
use crate::kernel_predicates::{DefaultKernelPredicateEvaluator, KernelPredicateEvaluator as _};
use crate::schema::{
    column_name, ColumnName, ColumnNamesAndTypes, DataType, SchemaRef, StructField, StructType,
};
use crate::table_configuration::TableConfiguration;
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
    CheckConstraints(constraints.into())
}

/// A fingerprint of everything a CHECK-constraint validation depends on: the constraint set (the
/// `delta.constraints.*` configuration entries), the logical schema, and the partition columns.
///
/// Two table states with equal fingerprints classify, parse, and evaluate every constraint
/// identically, so data validated against one is still valid against the other. On a commit
/// conflict a connector compares its transaction's fingerprint (what its data was validated
/// against) with the rebased snapshot's fingerprint: equal means it may retry the commit *without*
/// re-validating (a fast retry); unequal means the constraints must be re-checked against the new
/// table state first.
///
/// Schema and partition columns are part of the fingerprint, not just the SQL: the same constraint
/// text can classify or parse differently under a changed schema (a referenced column dropped or
/// type-widened) or a changed partition spec (data-only vs partition-only vs mixed), so comparing
/// the SQL alone would be unsound. Unrelated configuration changes (other table properties) are
/// excluded, so they never force a needless re-validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckConstraintFingerprint {
    /// Constraint name -> raw SQL. A `BTreeMap` so equality is independent of configuration order.
    constraints: BTreeMap<String, String>,
    schema: SchemaRef,
    partition_columns: Vec<String>,
}

impl CheckConstraintFingerprint {
    /// Builds a fingerprint from a table's configuration, logical schema, and partition columns.
    fn new(
        configuration: &HashMap<String, String>,
        schema: SchemaRef,
        partition_columns: &[String],
    ) -> Self {
        let constraints = configuration
            .iter()
            .filter_map(|(key, sql)| {
                strip_constraint_prefix(key).map(|name| (name.to_string(), sql.clone()))
            })
            .collect();
        Self {
            constraints,
            schema,
            partition_columns: partition_columns.to_vec(),
        }
    }

    /// Builds a fingerprint from a [`TableConfiguration`] -- a snapshot's or a transaction's.
    pub(crate) fn from_table_configuration(table_config: &TableConfiguration) -> Self {
        Self::new(
            table_config.metadata().configuration(),
            table_config.logical_schema(),
            table_config.partition_columns(),
        )
    }
}

/// All CHECK constraints on a table. Dereferences to a slice for per-constraint access.
///
/// The first question a connector asks is set-level, so it is answered here:
/// [`is_kernel_parsable`](Self::is_kernel_parsable) reports whether kernel parsed *every*
/// constraint. If it did, kernel enforces them all (per data batch or per partitioned write
/// context) and the connector proceeds normally. If not, the connector must evaluate the
/// remaining raw SQL itself -- [`connector_enforced`](Self::connector_enforced) yields exactly
/// those constraints -- or fail the write.
// `Arc<[_]>` (not `Vec`) so cloning the set is an O(1) refcount bump -- the transaction caches one
// parse and hands out cheap clones to discovery and the write path (see
// `constraints_from_configuration`).
#[derive(Debug, Clone, Default)]
pub struct CheckConstraints(Arc<[CheckConstraint]>);

impl CheckConstraints {
    /// True if kernel parsed every constraint, i.e. no constraint requires
    /// [`Connector`](CheckConstraintEnforcement::Connector) enforcement. Kernel then enforces all
    /// of them when a validator is built from the write context (automatic in the default engine's
    /// `write_parquet`): data-batch and partition+data
    /// ([`DataAndPartitionValues`](CheckConstraintEnforcement::DataAndPartitionValues)) constraints
    /// per batch, and partition-column
    /// ([`PartitionValues`](CheckConstraintEnforcement::PartitionValues)) constraints once, against
    /// the write context's partition values. The latter two need partition values, so the bare
    /// [`validator`](Self::validator) fails closed on them -- enforce them via
    /// `WriteContext::check_constraint_validator` / `validate_check_constraints`.
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

    /// Binds the kernel-evaluable constraints to `evaluation_handler` for per-batch validation;
    /// equivalent to [`CheckConstraintValidator::try_new`]. Fails closed if any constraint is
    /// connector-enforced, or depends on partition values (the
    /// [`PartitionValues`](CheckConstraintEnforcement::PartitionValues) and
    /// [`DataAndPartitionValues`](CheckConstraintEnforcement::DataAndPartitionValues) kinds) --
    /// those need a write context, so build one via `WriteContext::check_constraint_validator`
    /// instead. Check [`is_kernel_parsable`](Self::is_kernel_parsable) first and handle the
    /// [`connector_enforced`](Self::connector_enforced) remainder to avoid the connector error.
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
fn enforce_on_partition_values(
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
    /// The constraint references a partition column, so kernel evaluates it against the write
    /// context's partition values (per-file constants; readers reconstruct partition columns from
    /// `add.partitionValues`) -- once, when a validator is built from a partitioned write context
    /// (e.g. the default engine's `write_parquet`). Connectors that do not build kernel validators
    /// must enforce the constraint against their own partition values.
    PartitionValues,
    /// The constraint references both partition and data columns. Kernel evaluates it per data
    /// batch via a [`CheckConstraintValidator`], but only one built from a partitioned write
    /// context (`WriteContext::check_constraint_validator`): the write context supplies the
    /// partition values kernel overlays onto each batch. The per-constraint
    /// [`CheckConstraint::validate`], and a validator built without partition values, fail closed.
    DataAndPartitionValues,
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
    /// The expression references BOTH partition and data columns. Kernel evaluates it per data
    /// batch with the partition columns overlaid as their (per-file constant) write-context scalar
    /// values, so it can only be enforced through a partitioned write context (which supplies
    /// them).
    DataAndPartition(PredicateRef),
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
                    // References BOTH a partition column and a data column. No single source sees
                    // all the columns, so kernel evaluates it per data batch with the partition
                    // columns overlaid as their (per-file constant) write-context scalar values --
                    // which requires a partitioned write context to supply them.
                    Some(_) if references_data_column => {
                        ConstraintSupport::DataAndPartition(predicate)
                    }
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
            ConstraintSupport::DataAndPartition(_) => {
                CheckConstraintEnforcement::DataAndPartitionValues
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
            ConstraintSupport::DataAndPartition(predicate) => Some(predicate),
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
    ///   against the write context's partition values (build a validator from it), not against
    ///   batches, and connector-enforced constraints fail closed (see [`Self::enforcement`]).
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
                 against the write context's partition values when a validator is built from it \
                 (e.g. WriteContext::check_constraint_validator / validate_check_constraints), not \
                 against data batches passed here",
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

    /// The fail-closed error for a constraint that depends on partition values when no partitioned
    /// write context (hence no partition values) is available to evaluate it against. Covers both
    /// partition-only ([`ConstraintSupport::PartitionValues`]) and mixed
    /// ([`ConstraintSupport::DataAndPartition`]) constraints.
    fn partition_context_required_error(&self) -> Error {
        Error::unsupported(format!(
            "CHECK constraint '{}' ({}) depends on partition values; it can only be evaluated \
             through a partitioned write context (e.g. WriteContext::check_constraint_validator / \
             validate_check_constraints), which supplies them",
            self.name, self.raw_sql
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
    /// For [`ConstraintSupport::DataAndPartition`]: overlays the partition columns onto each batch
    /// (as their write-context scalar values) before `evaluator` runs. `None` for a plain
    /// data-batch constraint, which evaluates directly over the batch.
    augment: Option<Arc<dyn ExpressionEvaluator>>,
    values_renderer: Option<ViolationValuesRenderer>,
}

/// The table's kernel-evaluable CHECK constraints bound to an engine's [`EvaluationHandler`],
/// ready to validate batches. Build it once per write and reuse it for every batch: construction
/// surfaces connector-enforced constraints immediately (fail closed, before any data is written)
/// and amortizes predicate-evaluator creation across batches.
///
/// Partition-value-enforced constraints are evaluated once at construction, against the write
/// context's partition values; a validator built without them (the bare [`Self::try_new`]) fails
/// closed on them. Connectors that do not build kernel validators must enforce such constraints
/// against their own partition values.
///
/// Custom engines obtain one from `WriteContext::check_constraint_validator`, or directly from
/// the constraints returned by `Transaction::check_constraints`.
pub struct CheckConstraintValidator {
    bound: Vec<BoundConstraint>,
}

/// Builds an expression evaluator that overlays a write context's partition columns onto a logical
/// batch: each partition column is replaced by its (per-file constant) scalar value, every other
/// column passes through unchanged. This lets a predicate referencing both partition and data
/// columns evaluate over a single batch -- against the value kernel records in
/// `add.partitionValues` (the authoritative one), not whatever a batch might carry for the
/// partition column.
fn build_partition_augment(
    schema: &SchemaRef,
    partition_values: &HashMap<String, Scalar>,
    evaluation_handler: &dyn EvaluationHandler,
) -> DeltaResult<Arc<dyn ExpressionEvaluator>> {
    let fields: Vec<Expression> = schema
        .fields()
        .map(|field| match partition_values.get(field.name()) {
            Some(scalar) => Expression::literal(scalar.clone()),
            None => Expression::column([field.name()]),
        })
        .collect();
    evaluation_handler.new_expression_evaluator(
        schema.clone(),
        Arc::new(Expression::struct_from(fields)),
        DataType::from(schema.as_ref().clone()),
    )
}

impl CheckConstraintValidator {
    /// Binds `constraints` to `evaluation_handler`, erroring (fail closed) if any constraint is
    /// connector-enforced, or depends on partition values (the partition-only and partition+data
    /// kinds need a write context's partition values, which this constructor does not supply). A
    /// connector that enforces such constraints with its own SQL engine should filter them out and
    /// bind only the remainder.
    pub fn try_new(
        constraints: &[CheckConstraint],
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<Self> {
        Self::try_new_for_write_context(constraints, &HashMap::new(), evaluation_handler)
    }

    /// Like [`Self::try_new`], but supplied with the (logical, schema-cased) partition values of a
    /// write context. A constraint referencing both partition and data columns
    /// ([`ConstraintSupport::DataAndPartition`]) is bound by overlaying those partition values onto
    /// each batch; given an empty map (no write context), such a constraint fails closed.
    pub(crate) fn try_new_for_write_context(
        constraints: &[CheckConstraint],
        partition_values: &HashMap<String, Scalar>,
        evaluation_handler: &dyn EvaluationHandler,
    ) -> DeltaResult<Self> {
        // Partition-only constraints need no data batch: enforce them here, once, against the
        // write context's partition values (the per-batch loop below skips them). With a write
        // context (non-empty map) this is the partition-value enforcement point; without one, the
        // loop fails them closed, mirroring DataAndPartition.
        if !partition_values.is_empty() {
            enforce_on_partition_values(constraints, partition_values)?;
        }
        let mut bound = Vec::new();
        for constraint in constraints {
            let (predicate, augment) = match &constraint.support {
                ConstraintSupport::DataBatches(predicate) => (predicate, None),
                ConstraintSupport::DataAndPartition(predicate) => {
                    if partition_values.is_empty() {
                        return Err(constraint.partition_context_required_error());
                    }
                    let augment = build_partition_augment(
                        &constraint.schema,
                        partition_values,
                        evaluation_handler,
                    )?;
                    (predicate, Some(augment))
                }
                // Enforced above (once) against the write context's partition values, not per
                // batch. Without a write context (empty map), kernel cannot evaluate it -> fail
                // closed, mirroring DataAndPartition.
                ConstraintSupport::PartitionValues { .. } => {
                    if partition_values.is_empty() {
                        return Err(constraint.partition_context_required_error());
                    }
                    continue;
                }
                ConstraintSupport::Unsupported(_) => {
                    return Err(constraint.connector_enforced_error())
                }
            };
            bound.push(BoundConstraint {
                name: constraint.name.clone(),
                raw_sql: constraint.raw_sql.clone(),
                evaluator: evaluation_handler
                    .new_predicate_evaluator(constraint.schema.clone(), predicate.clone())?,
                augment,
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
            // Partition+data constraints overlay the partition scalars onto the batch first, so
            // the predicate (and the value renderer) see the authoritative partition value on
            // every row. Plain data-batch constraints evaluate directly over `batch`.
            let augmented = match &constraint.augment {
                Some(augment) => Some(augment.evaluate(batch)?),
                None => None,
            };
            let input: &dyn EngineData = augmented.as_deref().unwrap_or(batch);

            let result = constraint.evaluator.evaluate(input)?;
            let mut visitor = CheckResultVisitor::default();
            visitor.visit_rows_of(result.as_ref())?;
            let Some(violation) = visitor.violation else {
                continue;
            };
            let values = constraint
                .values_renderer
                .as_ref()
                .and_then(|renderer| renderer.render(input, violation.row))
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
    fn fingerprint_tracks_constraints_schema_and_partition_columns() {
        let pc = vec!["name".to_string()];
        let base = CheckConstraintFingerprint::new(
            &config(&[("delta.constraints.c", "amount > 0")]),
            schema(),
            &pc,
        );
        // Identical inputs -> equal (a fast retry is sound).
        assert_eq!(
            base,
            CheckConstraintFingerprint::new(
                &config(&[("delta.constraints.c", "amount > 0")]),
                schema(),
                &pc,
            )
        );
        // An unrelated table-property change is excluded -> still equal (no needless
        // re-validation).
        assert_eq!(
            base,
            CheckConstraintFingerprint::new(
                &config(&[
                    ("delta.constraints.c", "amount > 0"),
                    ("delta.appendOnly", "true")
                ]),
                schema(),
                &pc,
            )
        );
        // Changed SQL -> differ.
        assert_ne!(
            base,
            CheckConstraintFingerprint::new(
                &config(&[("delta.constraints.c", "amount > 5")]),
                schema(),
                &pc,
            )
        );
        // Added constraint -> differ.
        assert_ne!(
            base,
            CheckConstraintFingerprint::new(
                &config(&[
                    ("delta.constraints.c", "amount > 0"),
                    ("delta.constraints.d", "amount < 100"),
                ]),
                schema(),
                &pc,
            )
        );
        // Changed partition columns -> differ (data-only vs partition-only vs mixed
        // classification).
        assert_ne!(
            base,
            CheckConstraintFingerprint::new(
                &config(&[("delta.constraints.c", "amount > 0")]),
                schema(),
                &[]
            )
        );
        // Changed schema (amount LONG -> INTEGER) -> differ (parsing/coercion can change).
        let retyped = Arc::new(StructType::new_unchecked([
            StructField::nullable("amount", DataType::INTEGER),
            StructField::nullable("name", DataType::STRING),
        ]));
        assert_ne!(
            base,
            CheckConstraintFingerprint::new(
                &config(&[("delta.constraints.c", "amount > 0")]),
                retyped,
                &pc
            )
        );
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
    fn single_comparison_mixing_partition_and_data_columns_is_partition_augmented() {
        let partition_columns = vec!["name".to_string()];
        // A single comparison referencing both the partition column (`name`) and a data column
        // (`amount`): kernel evaluates it per batch with the partition value overlaid, so it is
        // partition-augmented (kernel-parsable) rather than connector-enforced.
        let mixed = CheckConstraint::new("mixed", "name != amount", schema(), &partition_columns);
        assert_eq!(
            mixed.enforcement(),
            CheckConstraintEnforcement::DataAndPartitionValues
        );
        assert!(mixed.predicate().is_some());

        // A junction is still outside the simple-comparison grammar, so it is connector-enforced
        // even though it also mixes a partition and a data column.
        let junction = CheckConstraint::new(
            "junction",
            "name = 'a' AND amount > 0",
            schema(),
            &partition_columns,
        );
        assert_eq!(
            junction.enforcement(),
            CheckConstraintEnforcement::Connector
        );

        // With only the partition-augmented mixed constraint, the table is fully kernel-parsable.
        let constraints = constraints_from_configuration(
            &config(&[("delta.constraints.mixed", "name != amount")]),
            schema(),
            &partition_columns,
        );
        assert!(constraints.is_kernel_parsable());
        assert_eq!(constraints.connector_enforced().count(), 0);
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
