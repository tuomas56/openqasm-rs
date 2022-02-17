# openqasm-rs

[![](https://img.shields.io/crates/v/openqasm)](https://crates.io/crates/openqasm) ![](https://img.shields.io/crates/l/openqasm.svg) [![](https://img.shields.io/docsrs/openqasm)](https://docs.rs/openqasm)

This crate implements a parser, type-checker and translator for OpenQASM 2.0. 

### Features

* Full type-checking both before and during translation.
* Beautiful error messages courtesy of [ariadne](https://crates.io/crates/ariadne).
* Pretty-printing with [pretty](https://crates.io/crates/pretty).
* Flexible include statement handling for sandboxing.
* Slightly relaxed from the specification for ease of use, for instance allowing gate definitions out of order and handling include-cycles gracefully.
* No unsafe code.

### Future Roadmap

* Support OpenQASM 3.0.
* Provide a syntax highlighting and language server extension.
* Provide a utility to visualize circuits.
* Transpile OpenQASM between versions 2.0 to 3.0, or into other languages like Quil.

### Examples

Parse a file and pretty print it:
```rust
use openqasm as oq;
use oq::GenericError;

fn main() {
    let mut cache = oq::SourceCache::new();
    let mut parser = oq::Parser::new(&mut cache)
        .with_file_policy(oq::parser::FilePolicy::Ignore);
    parser.parse_file("file.qasm");

    let prog = parser.done().to_errors().unwrap();
    println!("{}", prog.to_pretty(70));
}
```

Typecheck a program:
```rust
use openqasm as oq;
use oq::GenericError;

fn example(path: &str, cache: &mut oq::SourceCache) -> Result<(), oq::Errors> {
    let mut parser = oq::Parser::new(cache);
    parser.parse_file(path);
    let program = parser.done().to_errors()?;
    program.type_check().to_errors()?;
    Ok(())
}

fn main() {
    let mut cache = oq::SourceCache::new();
    if let Err(errors) = example("filename.qasm", &mut cache) {
        errors.print(&mut cache).unwrap();
    }
}
```

More examples are provided in the `examples` directory.
