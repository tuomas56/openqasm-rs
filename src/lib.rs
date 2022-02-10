//! This crate implements a parser, type-checker and translator for OpenQASM 2.0.
//!
//! ### Parsing
//!
//! The main interface for parsing is the `parser::Parser` struct, and this
//! produces either a `ast::Program` or a list of errors.
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
//! match parser.done() {
//!     Ok(program) => ..., // do something with this
//!     Err(errors) => for error in errors {
//!         // print the error to stderr
//!         error.eprint(&mut cache).unwrap();
//!     }
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
//! if let Err(errors) = program.type_check() {
//!     for error in errors {
//!         error.eprint(&mut cache).unwrap();
//!     }
//! }
//! ```

#![deny(mutable_borrow_reservation_conflict)]

#[macro_use]
extern crate lalrpop_util;

pub mod ast;
pub mod parser;
mod typing;

pub use ast::{Decl, Expr, Program, Reg, Stmt};
pub use parser::{Parser, SourceCache};
