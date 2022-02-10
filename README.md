# openqasm-rs

This crate implements a parser, type-checker and translator for OpenQASM 2.0. 

### Features

* Full type-checking both before and during translation.
* Robust and fairly fast parsing with [lalrpop](https://crates.io/crates/lalrpop).
* Beautiful error messages courtesy of [ariadne](https://crates.io/crates/ariadne).
* Flexible include statement handling for sandboxing.
* Slightly relaxed from the specification for ease of use, for instance allowing gate definitions out of order and handling include-cycles gracefully.
* No unsafe code.

### Future Roadmap

* Support OpenQASM 3.0.
* Provide a utility to visualize circuits.
* Transpile OpenQASM between versions 2.0 to 3.0, or into other languages like Quil.
