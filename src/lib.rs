//! This crate implements a parser, type-checker and translator for OpenQASM 2.0.
//!
//! ### Parsing
//!
//! The main interface for parsing is the `parser::Parser` struct, and this
//! produces either a `ast::Program` or a list of `parser::ParseError`s.
//! Example Usage:
//! ```rust
//! let mut cache = SourceCache::new();
//! let mut parser = Parser::new(&mut cache);
//! parser.parse_file("test.qasm");
//! parser.parse_source("
//!     OPENQASM 2.0;
//!     qreg a;
//!     creg b;
//!     cx a, b;
//! ");
//!
//! match parser.done().to_errors() {
//!     Ok(program) => ..., // do something with this
//!     Err(errors) => errors.print(&mut cache).unwrap()
//! }
//! ```
//!
//! ### Type-Checking
//!
//! Once you have a `Program`, you can type-check it with the
//! `ast::Program::type_check` method. This detects many types of
//! errors before translation.
//!
//! Example Usage:
//! ```rust
//! let mut cache = SourceCache::new();
//! let program: Program = ...; // obtain a program somehow
//!
//! if let Err(errors) = program.type_check().to_errors() {
//!     errors.print(&mut cache).unwrap();
//! }
//! ```
//!
//! ### Error Handling
//!
//! Specific error types for each process (parsing, checking etc.)
//! are provided. However, if you don't need to handle these individually,
//! you can use `GenericError` trait and `GenericError::to_errors` to convert
//! and `Result<_, SomeSpecificError>` into a generic `Result<_, Errors>` type.
//! This can then be handled, printed or converted to a `Report` as detailed below.
//!
//! ### Features
//!
//! The `ariadne` feature is enabled by default and allows
//! pretty-printing of errors using the `ariadne` crate by providing
//! `to_report` functions on all error types, as well as `Errors::eprint`/`print`.
//!
//! The `pretty` feature is enabled by default and allows pretty-printing of
//! AST objects using the `pretty` crate. This will implement the `pretty::Pretty`
//! trait on these objects, and also provides the `to_pretty` method to easily render
//! to a string.
//!

#![deny(mutable_borrow_reservation_conflict)]

#[macro_use]
extern crate lalrpop_util;

pub mod ast;
pub mod parser;
pub mod typing;

//#[cfg(feature = "pretty")]
mod pretty;

use thiserror::Error;

/// A generic error type for this crate.
/// This collects all the error types used in this crate
/// into one enum, for generic handling.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] parser::ParseError),
    #[error(transparent)]
    TypeError(#[from] typing::TypeError),
}

impl Error {
    /// Convert this error to a `Report` for printing.
    #[cfg(feature = "ariadne")]
    pub fn to_report(&self) -> ast::Report {
        match self {
            Error::ParseError(e) => e.to_report(),
            Error::TypeError(e) => e.to_report(),
        }
    }
}

/// Represents a collection of generic errors.
///
/// This struct contains a vector of `Error` types
/// that represent errors that have occured.
///
/// If `ariadne` is enabled, you can use `Errors::eprint`/`print`
/// to print all the errors to standard output.
#[derive(Debug, Error)]
pub struct Errors {
    pub errors: Vec<Error>,
}

impl std::fmt::Display for Errors {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for err in &self.errors {
            write!(f, "{}, ", err)?;
        }
        Ok(())
    }
}

#[cfg(feature = "ariadne")]
impl Errors {
    /// Convert all errors into `Report`s.
    pub fn as_reports(&self) -> impl Iterator<Item = ast::Report> + '_ {
        self.errors.iter().map(|err| err.to_report())
    }

    /// Print all the errors to stderr with `ariadne`.
    pub fn eprint(&self, cache: &mut parser::SourceCache) -> std::io::Result<()> {
        for report in self.as_reports() {
            report.eprint(&mut *cache)?;
            eprintln!();
        }
        Ok(())
    }

    /// Print all the errors to stdout with `ariadne`.
    pub fn print(&self, cache: &mut parser::SourceCache) -> std::io::Result<()> {
        for report in self.as_reports() {
            report.print(&mut *cache)?;
            println!();
        }
        Ok(())
    }

    /// Write all the errors to a `Write` implementation with `ariadne`.
    pub fn write<W: std::io::Write>(
        &self,
        cache: &mut parser::SourceCache,
        mut out: W,
    ) -> std::io::Result<()> {
        for report in self.as_reports() {
            report.write(&mut *cache, &mut out)?;
            writeln!(&mut out, "")?;
        }
        Ok(())
    }
}

/// A trait to convert a specific result type into a generic one.
/// This lets you convert `Result<T, E>` or `Result<T, Vec<E>>` into
/// `Result<T, Errors>` for any error type `E` defined in this library.
/// In this way you can ignore the specific type of error when you don't
/// need to handle them explicitly (which is most of the time).
///
/// Example Usage:
/// ```rust
/// fn test() -> Result<(), Errors> {
///     ...
///     let program = parser.done().to_errors()?;
///     program.type_check().to_errors()?;
/// }
/// ```
pub trait GenericError: Sized {
    type Inner;

    /// Convert any errors in this type to generic errors.
    fn to_errors(self) -> Result<Self::Inner, Errors>;
}

impl<T, E> GenericError for std::result::Result<T, Vec<E>>
where
    Error: From<E>,
{
    type Inner = T;

    fn to_errors(self) -> Result<T, Errors> {
        self.map_err(|errs| Errors {
            errors: errs.into_iter().map(Error::from).collect(),
        })
    }
}

impl<T, E> GenericError for std::result::Result<T, E>
where
    Error: From<E>,
{
    type Inner = T;

    fn to_errors(self) -> Result<T, Errors> {
        self.map_err(|e| Errors {
            errors: vec![Error::from(e)],
        })
    }
}

pub use ast::{Decl, Expr, Program, Reg, Stmt};
pub use parser::{Parser, SourceCache};
