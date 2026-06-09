//! Recursive-descent + precedence parser for CHECK-constraint predicates.
//!
//! Consumes the [`Token`] stream from [`super::token`] and produces an [`Ast`]. Precedence, from
//! lowest to highest, is `OR < AND < NOT < comparison`. Lowering ([`super::lower`]) resolves the
//! resulting tree against the table schema.

// WIP feature behind `check-constraints-in-dev`; some items have no caller until enforcement lands.
#![allow(dead_code)]

use super::token::Token;
use crate::{DeltaResult, Error};

/// An operand of a comparison: a column reference (raw path segments, as written) or a literal
/// (raw source text). Both are resolved/typed during lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Operand {
    Column(Vec<String>),
    Literal(String),
}

/// A comparison operator. Kernel has no native `<=`/`>=`/`!=`; lowering maps these to the
/// NOT-wrapping `Predicate` constructors (`le`/`ge`/`ne`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A parsed predicate tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Ast {
    Compare(CmpOp, Operand, Operand),
    IsNull {
        operand: Operand,
        negated: bool,
    },
    And(Box<Ast>, Box<Ast>),
    Or(Box<Ast>, Box<Ast>),
    Not(Box<Ast>),
    /// A bare boolean operand used directly as a predicate (a boolean column, `TRUE`, or `FALSE`).
    Operand(Operand),
}

/// Parse a full token stream into an [`Ast`], erroring if any tokens are left over.
pub(super) fn parse(tokens: Vec<Token>) -> DeltaResult<Ast> {
    let mut parser = Parser { tokens, pos: 0 };
    let ast = parser.parse_or()?;
    if parser.pos != parser.tokens.len() {
        return Err(Error::generic(
            "unexpected trailing input in CHECK constraint",
        ));
    }
    Ok(ast)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Consume `expected`, or error if the next token differs.
    fn expect(&mut self, expected: &Token) -> DeltaResult<()> {
        match self.advance() {
            Some(ref token) if token == expected => Ok(()),
            other => Err(Error::generic(format!(
                "expected {expected:?} in CHECK constraint, found {other:?}"
            ))),
        }
    }

    // or := and ( OR and )*
    fn parse_or(&mut self) -> DeltaResult<Ast> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Ast::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // and := not ( AND not )*
    fn parse_and(&mut self) -> DeltaResult<Ast> {
        let mut left = self.parse_not()?;
        while self.peek() == Some(&Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = Ast::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // not := NOT not | comparison
    fn parse_not(&mut self) -> DeltaResult<Ast> {
        if self.peek() == Some(&Token::Not) {
            self.advance();
            return Ok(Ast::Not(Box::new(self.parse_not()?)));
        }
        self.parse_comparison()
    }

    // comparison := '(' or ')'
    //             | operand ( cmp operand | IS NOT? NULL )?
    fn parse_comparison(&mut self) -> DeltaResult<Ast> {
        if self.peek() == Some(&Token::LParen) {
            self.advance();
            let inner = self.parse_or()?;
            self.expect(&Token::RParen)?;
            return Ok(inner);
        }
        let left = self.parse_operand()?;
        match self.peek() {
            Some(Token::Lt | Token::Le | Token::Gt | Token::Ge | Token::Eq | Token::Ne) => {
                let op = self.cmp_op()?;
                let right = self.parse_operand()?;
                Ok(Ast::Compare(op, left, right))
            }
            Some(Token::Is) => {
                self.advance();
                let negated = self.peek() == Some(&Token::Not);
                if negated {
                    self.advance();
                }
                self.expect(&Token::Null)?;
                Ok(Ast::IsNull {
                    operand: left,
                    negated,
                })
            }
            // A bare operand (boolean column / TRUE / FALSE) used as a predicate.
            _ => Ok(Ast::Operand(left)),
        }
    }

    fn cmp_op(&mut self) -> DeltaResult<CmpOp> {
        let op = match self.advance() {
            Some(Token::Lt) => CmpOp::Lt,
            Some(Token::Le) => CmpOp::Le,
            Some(Token::Gt) => CmpOp::Gt,
            Some(Token::Ge) => CmpOp::Ge,
            Some(Token::Eq) => CmpOp::Eq,
            Some(Token::Ne) => CmpOp::Ne,
            other => {
                return Err(Error::generic(format!(
                    "expected a comparison operator in CHECK constraint, found {other:?}"
                )))
            }
        };
        Ok(op)
    }

    // operand := Literal | NULL | Ident ( '.' Ident )*
    fn parse_operand(&mut self) -> DeltaResult<Operand> {
        match self.advance() {
            Some(Token::Literal(raw)) => Ok(Operand::Literal(raw)),
            Some(Token::Null) => Ok(Operand::Literal("NULL".to_string())),
            Some(Token::Ident(first)) => {
                let mut path = vec![first];
                while self.peek() == Some(&Token::Dot) {
                    self.advance();
                    match self.advance() {
                        Some(Token::Ident(segment)) => path.push(segment),
                        other => {
                            return Err(Error::generic(format!(
                            "expected an identifier after '.' in CHECK constraint, found {other:?}"
                        )))
                        }
                    }
                }
                Ok(Operand::Column(path))
            }
            other => Err(Error::generic(format!(
                "expected a column or literal in CHECK constraint, found {other:?}"
            ))),
        }
    }
}
