extern crate lalrpop;
extern crate rustc_version;

fn main() {
    match rustc_version::version_meta().unwrap().channel {
        rustc_version::Channel::Nightly => {
            println!("cargo:rustc-cfg=nightly_build");
        }
        _ => (),
    }

    lalrpop::process_root().unwrap();
}
