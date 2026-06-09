//! Tokenizer for the CHECK-constraint predicate parser.
//!
//! Splits a constraint SQL string into [`Token`]s. Literal tokens keep their *raw* source text
//! (quotes and typed-literal keyword included) so lowering can hand them to the existing literal
//! parser [`super::parse_sql`] rather than re-implementing scalar parsing here.

// WIP feature behind `check-constraints-in-dev`; some items have no caller until enforcement lands.
#![allow(dead_code)]

use std::iter::Peekable;
use std::str::Chars;

use crate::{DeltaResult, Error};

/// A peekable character stream over the constraint source string.
type CharStream<'a> = Peekable<Chars<'a>>;

/// A lexical token in a CHECK-constraint predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Token {
    /// A single identifier segment (one component of a column path). Dotted paths are assembled by
    /// the parser from `Ident (Dot Ident)*`.
    Ident(String),
    /// The raw source text of a literal -- a number, a `'string'` (quotes retained), an `X'..'`
    /// binary, `TRUE`/`FALSE`, or a typed `DATE '...'` / `TIMESTAMP '...'` / `TIMESTAMP_NTZ '...'`
    /// literal. Decoding is deferred to [`super::parse_sql`].
    Literal(String),
    And,
    Or,
    Not,
    Is,
    Null,
    Dot,
    LParen,
    RParen,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    /// `<>` or `!=` (both lower to the same not-equal predicate).
    Ne,
}

/// Tokenize a constraint expression, or return an error for any character/run the grammar does not
/// recognize (e.g. arithmetic operators, an unterminated string).
pub(super) fn tokenize(sql: &str) -> DeltaResult<Vec<Token>> {
    let mut chars = sql.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            '.' => {
                chars.next();
                tokens.push(Token::Dot);
            }
            '=' => {
                chars.next();
                tokens.push(Token::Eq);
            }
            '<' => {
                chars.next();
                let tok = if chars.next_if_eq(&'=').is_some() {
                    Token::Le
                } else if chars.next_if_eq(&'>').is_some() {
                    Token::Ne
                } else {
                    Token::Lt
                };
                tokens.push(tok);
            }
            '>' => {
                chars.next();
                let tok = if chars.next_if_eq(&'=').is_some() {
                    Token::Ge
                } else {
                    Token::Gt
                };
                tokens.push(tok);
            }
            '!' => {
                chars.next();
                match chars.next_if_eq(&'=') {
                    Some(_) => tokens.push(Token::Ne),
                    None => return Err(unexpected('!', sql)),
                }
            }
            '\'' => tokens.push(Token::Literal(take_quoted_string(&mut chars, sql)?)),
            // A sign is only valid as the start of a numeric literal -- the grammar has no
            // arithmetic, so `a - b` is intentionally rejected.
            '+' | '-' => {
                let mut lookahead = chars.clone();
                lookahead.next();
                match lookahead.peek() {
                    Some(d) if d.is_ascii_digit() || *d == '.' => {
                        tokens.push(Token::Literal(take_number(&mut chars)))
                    }
                    _ => return Err(unexpected(c, sql)),
                }
            }
            c if c.is_ascii_digit() => tokens.push(Token::Literal(take_number(&mut chars))),
            c if c.is_ascii_alphabetic() || c == '_' => {
                tokens.push(classify_word(&mut chars, sql)?)
            }
            _ => return Err(unexpected(c, sql)),
        }
    }
    Ok(tokens)
}

fn unexpected(c: char, sql: &str) -> Error {
    Error::generic(format!(
        "unexpected character '{c}' in CHECK constraint: {sql}"
    ))
}

/// Consume a `'...'` string literal, returning its raw text *including* the surrounding quotes and
/// any doubled `''` escapes. Decoding (un-escaping) is left to [`super::parse_sql`]. The caller
/// guarantees a leading `'`.
fn take_quoted_string(chars: &mut CharStream<'_>, sql: &str) -> DeltaResult<String> {
    let unterminated = || {
        Error::generic(format!(
            "unterminated string literal in CHECK constraint: {sql}"
        ))
    };
    let mut out = String::new();
    match chars.next() {
        Some('\'') => out.push('\''),
        _ => return Err(unterminated()),
    }
    loop {
        match chars.next() {
            Some('\'') => {
                out.push('\'');
                // A doubled quote is an embedded single quote; keep scanning. Otherwise this was
                // the closing quote.
                match chars.next_if_eq(&'\'') {
                    Some(q) => out.push(q),
                    None => return Ok(out),
                }
            }
            Some(c) => out.push(c),
            None => return Err(unterminated()),
        }
    }
}

/// Consume a numeric literal: optional leading sign, integer/fraction digits, optional exponent.
/// Stops before a second operator so `1+1` is not swallowed as one number.
fn take_number(chars: &mut CharStream<'_>) -> String {
    let mut out = String::new();
    if let Some(sign) = chars.next_if(|c| *c == '+' || *c == '-') {
        out.push(sign);
    }
    while let Some(c) = chars.next_if(|c| c.is_ascii_digit() || *c == '.') {
        out.push(c);
    }
    if let Some(e) = chars.next_if(|c| *c == 'e' || *c == 'E') {
        out.push(e);
        if let Some(sign) = chars.next_if(|c| *c == '+' || *c == '-') {
            out.push(sign);
        }
        while let Some(c) = chars.next_if(|c| c.is_ascii_digit()) {
            out.push(c);
        }
    }
    out
}

/// Scan a bare word, then classify it as a keyword, a boolean/typed/binary literal, or an
/// identifier. Typed (`DATE '...'`) and binary (`X'..'`) literals are recognized here so their raw
/// text reaches [`super::parse_sql`] as a single token.
fn classify_word(chars: &mut CharStream<'_>, sql: &str) -> DeltaResult<Token> {
    let mut word = String::new();
    while let Some(c) = chars.next_if(|c| c.is_ascii_alphanumeric() || *c == '_') {
        word.push(c);
    }
    let token = match word.to_ascii_uppercase().as_str() {
        "AND" => Token::And,
        "OR" => Token::Or,
        "NOT" => Token::Not,
        "IS" => Token::Is,
        "NULL" => Token::Null,
        "TRUE" | "FALSE" => Token::Literal(word),
        // `X'deadbeef'` binary literal: only when a quote immediately follows (no whitespace),
        // otherwise `x` is a column named `x`.
        "X" if chars.peek() == Some(&'\'') => {
            let quoted = take_quoted_string(chars, sql)?;
            Token::Literal(format!("{word}{quoted}"))
        }
        // Typed literal: the keyword followed (whitespace allowed) by a quoted string. Without the
        // quote it is just a column named `date`/`timestamp`/etc.
        "DATE" | "TIMESTAMP" | "TIMESTAMP_NTZ" if quote_follows(chars) => {
            while chars.next_if(|c| c.is_whitespace()).is_some() {}
            let quoted = take_quoted_string(chars, sql)?;
            Token::Literal(format!("{word} {quoted}"))
        }
        _ => Token::Ident(word),
    };
    Ok(token)
}

/// Peek (without consuming) past any whitespace to see whether the next character is a `'`.
fn quote_follows(chars: &CharStream<'_>) -> bool {
    let mut lookahead = chars.clone();
    while matches!(lookahead.peek(), Some(c) if c.is_whitespace()) {
        lookahead.next();
    }
    lookahead.peek() == Some(&'\'')
}
