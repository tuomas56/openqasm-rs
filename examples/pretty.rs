use openqasm as oq;
use oq::GenericError;

fn main() {
    let mut cache = oq::SourceCache::new();
    let mut parser = oq::Parser::new(&mut cache).with_file_policy(oq::parser::FilePolicy::Ignore);
    parser.parse_file("examples/good.qasm");

    let prog = parser.done().to_errors().unwrap();
    println!("{}", prog.to_pretty(70));
}
