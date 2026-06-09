//! Lowering: resolve a parsed [`Ast`] against the table schema into a kernel [`Predicate`].
//!
//! Column references resolve case-insensitively (matching Delta/Spark and kernel's other
//! column-resolution paths). A literal's type is inferred from the column on the other side of its
//! comparison, and the literal itself is parsed by reusing [`super::parse_sql`].

// WIP feature behind `check-constraints-in-dev`; some items have no caller until enforcement lands.
#![allow(dead_code)]

use super::parse_sql;
use super::parser::{Ast, CmpOp, Operand};
use crate::expressions::{ColumnName, Expression, Predicate};
use crate::schema::{DataType, StructType};
use crate::{DeltaResult, Error};

/// Lower a parsed predicate AST into a kernel [`Predicate`].
pub(super) fn lower(ast: &Ast, schema: &StructType) -> DeltaResult<Predicate> {
    match ast {
        Ast::And(left, right) => Ok(Predicate::and(lower(left, schema)?, lower(right, schema)?)),
        Ast::Or(left, right) => Ok(Predicate::or(lower(left, schema)?, lower(right, schema)?)),
        Ast::Not(inner) => Ok(Predicate::not(lower(inner, schema)?)),
        Ast::IsNull { operand, negated } => {
            let (expr, _) = resolve_operand(operand, schema, None)?;
            Ok(if *negated {
                Predicate::is_not_null(expr)
            } else {
                Predicate::is_null(expr)
            })
        }
        Ast::Compare(op, left, right) => lower_compare(*op, left, right, schema),
        Ast::Operand(operand) => {
            // A bare operand used as a predicate must be boolean-valued.
            let (expr, ty) = resolve_operand(operand, schema, Some(&DataType::BOOLEAN))?;
            if matches!(ty, Some(t) if t != DataType::BOOLEAN) {
                return Err(Error::generic(
                    "CHECK constraint operand is not a boolean expression",
                ));
            }
            Ok(Predicate::from_expr(expr))
        }
    }
}

fn lower_compare(
    op: CmpOp,
    left: &Operand,
    right: &Operand,
    schema: &StructType,
) -> DeltaResult<Predicate> {
    let left_type = column_type(left, schema)?;
    let right_type = column_type(right, schema)?;
    if left_type.is_none() && right_type.is_none() {
        return Err(Error::generic(
            "CHECK constraint comparison must reference at least one column",
        ));
    }
    // A literal is typed by the column on the other side of the comparison.
    let (left_expr, _) = resolve_operand(left, schema, right_type.as_ref())?;
    let (right_expr, _) = resolve_operand(right, schema, left_type.as_ref())?;
    Ok(match op {
        CmpOp::Eq => Predicate::eq(left_expr, right_expr),
        CmpOp::Ne => Predicate::ne(left_expr, right_expr),
        CmpOp::Lt => Predicate::lt(left_expr, right_expr),
        CmpOp::Le => Predicate::le(left_expr, right_expr),
        CmpOp::Gt => Predicate::gt(left_expr, right_expr),
        CmpOp::Ge => Predicate::ge(left_expr, right_expr),
    })
}

/// The resolved leaf [`DataType`] for a column operand, or `None` for a literal.
fn column_type(operand: &Operand, schema: &StructType) -> DeltaResult<Option<DataType>> {
    match operand {
        Operand::Literal(_) => Ok(None),
        Operand::Column(path) => Ok(Some(resolve_column(path, schema)?.1)),
    }
}

/// Resolve an operand into an [`Expression`]. Columns also return their resolved leaf type;
/// literals are parsed against `type_hint` (the compared column's type) via [`parse_sql`] and
/// carry no independent type.
fn resolve_operand(
    operand: &Operand,
    schema: &StructType,
    type_hint: Option<&DataType>,
) -> DeltaResult<(Expression, Option<DataType>)> {
    match operand {
        Operand::Column(path) => {
            let (canonical, data_type) = resolve_column(path, schema)?;
            Ok((Expression::column(canonical), Some(data_type)))
        }
        Operand::Literal(raw) => {
            let data_type = type_hint.ok_or_else(|| {
                Error::generic(format!(
                    "cannot type literal '{raw}': a CHECK constraint comparison must reference a column"
                ))
            })?;
            Ok((parse_sql(raw, data_type)?, None))
        }
    }
}

/// Resolve a (case-insensitive) column path against `schema`, returning the *canonical* path (the
/// schema's stored field names) and the leaf field's type. The canonical names are what the engine
/// sees in the logical batch, so the emitted column reference must use them rather than the
/// as-written casing.
fn resolve_column(path: &[String], schema: &StructType) -> DeltaResult<(Vec<String>, DataType)> {
    let column = ColumnName::new(path.iter().cloned());
    let fields = schema.walk_column_fields_by(&column, |parent, name| {
        parent
            .fields()
            .find(|f| f.name().eq_ignore_ascii_case(name))
    })?;
    let canonical: Vec<String> = fields.iter().map(|f| f.name().to_string()).collect();
    let leaf = fields
        .last()
        .ok_or_else(|| Error::generic("CHECK constraint references an empty column path"))?;
    Ok((canonical, leaf.data_type().clone()))
}

#[cfg(test)]
mod tests {
    use super::super::parse_sql_predicate;
    use crate::expressions::{Expression, Predicate};
    use crate::schema::{DataType, StructField, StructType};

    fn schema() -> StructType {
        StructType::new_unchecked([
            StructField::nullable("amount", DataType::LONG),
            StructField::nullable("price", DataType::INTEGER),
            StructField::nullable("name", DataType::STRING),
            StructField::nullable("active", DataType::BOOLEAN),
        ])
    }

    fn col(name: &str) -> Expression {
        Expression::column([name])
    }

    #[test]
    fn parses_simple_comparison() {
        let pred = parse_sql_predicate("amount > 0", &schema()).unwrap();
        assert_eq!(
            pred,
            Predicate::gt(col("amount"), Expression::literal(0i64))
        );
    }

    #[test]
    fn types_literal_from_compared_column() {
        // `price` is INTEGER, so the literal must parse as Integer, not the default Long.
        let pred = parse_sql_predicate("price <= 10", &schema()).unwrap();
        assert_eq!(
            pred,
            Predicate::le(col("price"), Expression::literal(10i32))
        );
    }

    #[test]
    fn parses_string_equality_with_doubled_quote_escape() {
        let pred = parse_sql_predicate("name = 'O''Brien'", &schema()).unwrap();
        assert_eq!(
            pred,
            Predicate::eq(col("name"), Expression::literal("O'Brien"))
        );
    }

    #[test]
    fn parses_and_or_not_with_precedence() {
        let pred =
            parse_sql_predicate("amount > 0 AND price < 10 OR NOT active", &schema()).unwrap();
        let expected = Predicate::or(
            Predicate::and(
                Predicate::gt(col("amount"), Expression::literal(0i64)),
                Predicate::lt(col("price"), Expression::literal(10i32)),
            ),
            Predicate::not(Predicate::from_expr(col("active"))),
        );
        assert_eq!(pred, expected);
    }

    #[test]
    fn parentheses_override_precedence() {
        let pred = parse_sql_predicate("NOT (amount > 0 AND active)", &schema()).unwrap();
        let expected = Predicate::not(Predicate::and(
            Predicate::gt(col("amount"), Expression::literal(0i64)),
            Predicate::from_expr(col("active")),
        ));
        assert_eq!(pred, expected);
    }

    #[test]
    fn parses_is_not_null() {
        let pred = parse_sql_predicate("name IS NOT NULL", &schema()).unwrap();
        assert_eq!(pred, Predicate::is_not_null(col("name")));
    }

    #[test]
    fn resolves_columns_case_insensitively_to_canonical_name() {
        let pred = parse_sql_predicate("AMOUNT >= 0", &schema()).unwrap();
        assert_eq!(
            pred,
            Predicate::ge(col("amount"), Expression::literal(0i64))
        );
    }

    #[test]
    fn rejects_unknown_column() {
        assert!(parse_sql_predicate("nope > 0", &schema()).is_err());
    }

    #[test]
    fn rejects_unsupported_function_call() {
        assert!(parse_sql_predicate("length(name) > 0", &schema()).is_err());
    }

    #[test]
    fn rejects_arithmetic() {
        assert!(parse_sql_predicate("amount + 1 > 0", &schema()).is_err());
    }

    #[test]
    fn rejects_comparison_of_two_literals() {
        assert!(parse_sql_predicate("1 > 0", &schema()).is_err());
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(parse_sql_predicate("name = 'oops", &schema()).is_err());
    }
}
