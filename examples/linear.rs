use openqasm as oq;
use oq::translate::ProgramVisitor;
use oq::{
    ast::Symbol,
    translate::{GateWriter, Value},
    GenericError,
};

struct GatePrinter;

impl GateWriter for GatePrinter {
    fn initialize(&mut self, _: usize, _: usize) {}

    fn write_cx(&mut self, copy: usize, xor: usize) {
        println!("cx {copy} {xor}");
    }

    fn write_u(&mut self, theta: Value, phi: Value, lambda: Value, reg: usize) {
        println!("u({theta}, {phi}, {lambda}) {reg}");
    }

    fn write_opaque(&mut self, name: &Symbol, _: &[Value], _: &[usize]) {
        println!("opaque gate {}", name)
    }

    fn write_barrier(&mut self, _: &[usize]) {}

    fn write_measure(&mut self, from: usize, to: usize) {
        println!("measure {} -> {}", from, to);
    }

    fn write_reset(&mut self, reg: usize) {
        println!("reset {reg}");
    }

    fn start_conditional(&mut self, reg: usize, count: usize, value: usize) {
        println!("if ({reg}:{count} == {value}) {{");
    }

    fn end_conditional(&mut self) {
        println!("}}");
    }
}

fn example(path: &str, cache: &mut oq::SourceCache) -> Result<(), oq::Errors> {
    let mut parser = oq::Parser::new(cache);
    parser.parse_file(path);
    let program = parser.done().to_errors()?;
    program.type_check().to_errors()?;

    let mut l = oq::translate::Linearize::new(GatePrinter);
    l.visit_program(&program).unwrap();

    Ok(())
}

fn main() {
    let mut cache = oq::SourceCache::new();
    example("examples/good.qasm", &mut cache).unwrap();
}
