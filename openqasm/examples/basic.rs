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

    println!("Processing: good.qasm:");
    if let Err(errors) = example(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/good.qasm"), &mut cache) {
        errors.print(&mut cache).unwrap();
    }

    println!("Processing: bad.qasm:");
    if let Err(errors) = example(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bad.qasm"), &mut cache) {
        errors.print(&mut cache).unwrap();
    }
}
