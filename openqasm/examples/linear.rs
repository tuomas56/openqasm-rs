use openqasm as oq;
use oq::translate::ProgramVisitor;
use oq::{
    ast::Symbol,
    translate::{GateWriter, Value},
    GenericError,
};

struct GatePrinter;

impl GateWriter for GatePrinter {
    type Error = std::convert::Infallible;

    fn initialize(&mut self, _: &[Symbol], _: &[Symbol]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error> {
        println!("cx {copy} {xor}");
        Ok(())
    }

    fn write_u(&mut self, theta: Value, phi: Value, lambda: Value, reg: usize) -> Result<(), Self::Error> {
        println!("u({theta}, {phi}, {lambda}) {reg}");
        Ok(())
    }

    fn write_opaque(&mut self, name: &Symbol, _: &[Value], _: &[usize]) -> Result<(), Self::Error> {
        println!("opaque gate {}", name);
        Ok(())
    }

    fn write_barrier(&mut self, _: &[usize]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error> {
        println!("measure {} -> {}", from, to);
        Ok(())
    }

    fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error> {
        println!("reset {reg}");
        Ok(())
    }

    fn start_conditional(&mut self, reg: usize, count: usize, value: u64) -> Result<(), Self::Error> {
        println!("if ({reg}:{count} == {value}) {{");
        Ok(())
    }

    fn end_conditional(&mut self) -> Result<(), Self::Error> {
        println!("}}");
        Ok(())
    }
}

fn example(path: &str, cache: &mut oq::SourceCache) -> Result<(), oq::Errors> {
    let mut parser = oq::Parser::new(cache);
    parser.parse_file(path);
    let program = parser.done().to_errors()?;
    program.type_check().to_errors()?;

    let mut l = oq::translate::Linearize::new(GatePrinter);
    l.visit_program(&program).to_errors()?;

    Ok(())
}

fn main() {
    let mut cache = oq::SourceCache::new();
    if let Err(e) = example(
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/good.qasm"),
        &mut cache,
    ) {
        e.print(&mut cache).unwrap();
    }
}
