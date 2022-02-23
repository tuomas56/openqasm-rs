use crate::ast::{Decl, Expr, FileSpan, Program, Reg, Span, Stmt, Symbol};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use thiserror::Error;

#[cfg(feature = "ariadne")]
use {
    crate::ast::Report,
    ariadne::{Label, ReportKind},
};

mod value;
pub use value::Value;

/// A binary operation encountered in an `Expr`.
#[derive(Debug, Copy, Clone)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// An unary operation encountered in an `Expr`.
#[derive(Debug, Copy, Clone)]
pub enum Unop {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
    Neg,
}

/// Low-level translation interface for definitions and declarations.
///
/// This trait allows you to walk over the structure of `Program`
/// element by element. Each `visit_` method defines the callback
/// for what to do when a particular element is encountered.
///
/// Any unimplemented `visit_` methods have a default which traverses
/// all substructures in the element. Since this traversal does not happen
/// automatically, if you override a `visit_` method, you must either
/// traverse the element manually, or call the corresponding `walk_` method
/// which does the default traversal.
///
/// Also note that the declarations are garuanteed to be visited in the
/// following order by type: `Include`, `Def`, `QReg`, `CReg`, `Stmt`.
/// The error type is provided to allow you to fail out at any time by
/// returning an `Err` variant.
///
/// For example, to get the names of all gate definitions in a program,
/// you might do the following:
/// ```ignore
/// struct DefFinder;
///
/// impl ProgramVisitor for DefFinder {
///     fn visit_gate_def(
///         &mut self,
///         name: &Span<Symbol>,
///         params: &[Span<Symbol>],
///         args: &[Span<Symbol>],
///         body: &[Span<Stmt>]
///     ) {
///         println!("Found definition of `{}`.", name.inner);
///         self.walk_gate_def(name, params, args, body);
///     }
/// }
///
/// fn main() {
///     let program = ...; // aquire a program from somewhere
///     DefFinder.visit_program(&program);
/// }
/// ```
#[allow(unused_variables)]
pub trait ProgramVisitor {
    type Error;

    fn visit_program(&mut self, program: &Program) -> Result<(), Self::Error> {
        self.walk_program(program)
    }

    fn walk_program(&mut self, program: &Program) -> Result<(), Self::Error> {
        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::Include { .. }) {
                self.visit_decl(decl)?;
            }
        }

        for decl in &program.decls {
            let decl = decl;
            if matches!(&*decl.inner, Decl::Def { .. }) {
                self.visit_decl(decl)?;
            }
        }

        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::QReg { .. }) {
                self.visit_decl(decl)?;
            }
        }

        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::CReg { .. }) {
                self.visit_decl(decl)?;
            }
        }

        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::Stmt(..)) {
                self.visit_decl(decl)?;
            }
        }

        Ok(())
    }

    fn visit_decl(&mut self, decl: &Span<Decl>) -> Result<(), Self::Error> {
        self.walk_decl(decl)
    }

    fn walk_decl(&mut self, decl: &Span<Decl>) -> Result<(), Self::Error> {
        match &*decl.inner {
            Decl::Include { file } => self.visit_include(file),
            Decl::QReg { reg } => self.visit_qreg(reg),
            Decl::CReg { reg } => self.visit_creg(reg),
            Decl::Def {
                name,
                params,
                args,
                body,
            } => match body {
                None => self.visit_opaque_def(name, &params, &args),
                Some(body) => self.visit_gate_def(name, &params, &args, &body),
            },
            Decl::Stmt(stmt) => self.visit_stmt(stmt),
        }
    }

    fn visit_include(&mut self, file: &Span<Symbol>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_qreg(&mut self, reg: &Span<Reg>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_creg(&mut self, reg: &Span<Reg>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_opaque_def(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_gate_def(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
        body: &[Span<Stmt>],
    ) -> Result<(), Self::Error> {
        self.walk_gate_def(name, params, args, body)
    }

    fn walk_gate_def(
        &mut self,
        name: &Symbol,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
        body: &[Span<Stmt>],
    ) -> Result<(), Self::Error> {
        for stmt in body {
            self.visit_stmt(stmt)?
        }
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &Span<Stmt>) -> Result<(), Self::Error> {
        self.walk_stmt(stmt)
    }

    fn walk_stmt(&mut self, stmt: &Span<Stmt>) -> Result<(), Self::Error> {
        match &*stmt.inner {
            Stmt::Barrier { regs } => self.visit_barrier(&regs),
            Stmt::Measure { from, to } => self.visit_measure(from, to),
            Stmt::Reset { reg } => self.visit_reset(reg),
            Stmt::CX { copy, xor } => self.visit_cx(copy, xor),
            Stmt::U {
                theta,
                phi,
                lambda,
                reg,
            } => self.visit_u(theta, phi, lambda, reg),
            Stmt::Gate { name, params, args } => self.visit_gate(name, &params, &args),
            Stmt::Conditional { reg, val, then } => self.visit_conditional(reg, val, then),
        }
    }

    fn visit_barrier(&mut self, regs: &[Span<Reg>]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_measure(&mut self, from: &Span<Reg>, to: &Span<Reg>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_reset(&mut self, reg: &Span<Reg>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_cx(&mut self, copy: &Span<Reg>, xor: &Span<Reg>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_u(
        &mut self,
        theta: &Span<Expr>,
        phi: &Span<Expr>,
        lambda: &Span<Expr>,
        reg: &Span<Reg>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_gate(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Expr>],
        args: &[Span<Reg>],
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_conditional(
        &mut self,
        reg: &Span<Reg>,
        val: &Span<u64>,
        then: &Span<Stmt>,
    ) -> Result<(), Self::Error> {
        self.walk_conditional(reg, val, then)
    }

    fn walk_conditional(
        &mut self,
        reg: &Span<Reg>,
        val: &Span<u64>,
        then: &Span<Stmt>,
    ) -> Result<(), Self::Error> {
        self.visit_stmt(then)
    }
}

/// Companion trait to `ProgramVisitor`.
///
/// This visits an `Expr` piece by piece and uses the required
/// callbacks to compute a return value.
/// The type of the return value is given by `ExprVisitor::Output`.
///
/// The default `visit_` methods just traverse the structure and
/// call the callbacks where necessary, so if you override them,
/// you must do the traversal yourself.
///
/// For example, if you wanted to compute the maximum integer
/// value that appeared in an expression, you could do:
/// ```ignore
/// struct MaxFinder;
///
/// impl ExprVisitor for MaxFinder {
///     type Output = u64;
///
///     fn binop(&mut self, _: Binop, a: u64, b: u64) -> u64 { a.max(b) }
///     fn unop(&mut self, _: Unop, a: u64) -> u64 { a }
///     fn lookup(&mut self, _: &Symbol) -> u64 { 0 }
///     fn pi(&mut self) -> u64 { 0 }
///     fn int(&mut self, val: u64) -> u64 { val }
///     fn real(&mut self, val: f32) -> u64 { 0 }
/// }
///
/// fn main() {
///     let expr = ...; // acquire an expression from somewhere.
///     println!("max = {}", MaxFinder.visit_expr(&expr));
/// }
/// ```
pub trait ExprVisitor {
    type Output;

    fn binop(&mut self, op: Binop, a: Self::Output, b: Self::Output) -> Self::Output;
    fn unop(&mut self, op: Unop, a: Self::Output) -> Self::Output;
    fn lookup(&mut self, var: &Symbol) -> Self::Output;
    fn pi(&mut self) -> Self::Output;
    fn int(&mut self, val: u64) -> Self::Output;
    fn real(&mut self, val: f32) -> Self::Output;

    fn visit_expr(&mut self, expr: &Span<Expr>) -> Self::Output {
        match &*expr.inner {
            Expr::Pi => self.pi(),
            Expr::Int(val) => self.int(*val),
            Expr::Real(val) => self.real(*val),
            Expr::Var(symbol) => self.visit_var(symbol),
            Expr::Add(a, b) => self.visit_binop(Binop::Add, a, b),
            Expr::Sub(a, b) => self.visit_binop(Binop::Sub, a, b),
            Expr::Mul(a, b) => self.visit_binop(Binop::Mul, a, b),
            Expr::Div(a, b) => self.visit_binop(Binop::Div, a, b),
            Expr::Pow(a, b) => self.visit_binop(Binop::Pow, a, b),
            Expr::Sin(a) => self.visit_unop(Unop::Sin, a),
            Expr::Cos(a) => self.visit_unop(Unop::Cos, a),
            Expr::Tan(a) => self.visit_unop(Unop::Tan, a),
            Expr::Exp(a) => self.visit_unop(Unop::Exp, a),
            Expr::Ln(a) => self.visit_unop(Unop::Ln, a),
            Expr::Sqrt(a) => self.visit_unop(Unop::Sqrt, a),
            Expr::Neg(a) => self.visit_unop(Unop::Neg, a),
        }
    }

    fn visit_var(&mut self, var: &Symbol) -> Self::Output {
        self.lookup(var)
    }

    fn visit_binop(&mut self, op: Binop, a: &Span<Expr>, b: &Span<Expr>) -> Self::Output {
        let a = self.visit_expr(a);
        let b = self.visit_expr(b);
        self.binop(op, a, b)
    }

    fn visit_unop(&mut self, op: Unop, a: &Span<Expr>) -> Self::Output {
        let a = self.visit_expr(a);
        self.unop(op, a)
    }
}

struct FrameEvaluator<'a> {
    frame: &'a Frame,
}

// These are the possible errors that can occur
// when evaluating a parameter. Unfortunately, there
// are still some we don't catch, for example
// certain uses of constants that don't fit in a i64.
// Additionally, a lot of errors will just show up as
// `ApproximateFail(NaN)`
#[derive(Debug)]
enum EvalError {
    DivideByZero,
    ApproximateFail(f32),
    OverflowError,
}

impl<'a> ExprVisitor for FrameEvaluator<'a> {
    type Output = Result<Value, EvalError>;

    fn pi(&mut self) -> Self::Output {
        Ok(Value::PI)
    }

    fn int(&mut self, val: u64) -> Self::Output {
        Ok(Value::int(val as i64))
    }

    fn real(&mut self, val: f32) -> Self::Output {
        Value::from_float(val).ok_or(EvalError::ApproximateFail(val))
    }

    fn unop(&mut self, op: Unop, a: Self::Output) -> Self::Output {
        let a = a?;
        match op {
            Unop::Neg => a.neg_internal(),
            Unop::Sin => a.sin_internal(),
            Unop::Cos => a.cos_internal(),
            Unop::Tan => a.tan_internal(),
            Unop::Exp => a.exp_internal(),
            Unop::Ln => a.ln_internal(),
            Unop::Sqrt => a.sqrt_internal(),
        }
    }

    fn binop(&mut self, op: Binop, a: Self::Output, b: Self::Output) -> Self::Output {
        let (a, b) = (a?, b?);
        match op {
            Binop::Add => a.add_internal(b),
            Binop::Sub => a.sub_internal(b),
            Binop::Mul => a.mul_internal(b),
            Binop::Div => a.div_internal(b),
            Binop::Pow => a.pow_internal(b),
        }
    }

    fn lookup(&mut self, var: &Symbol) -> Self::Output {
        Ok(self.frame.params[var])
    }
}

struct Frame {
    name: Symbol,
    def_name: Symbol,
    call: Option<FileSpan>,
    qregs: HashMap<Symbol, (u64, u64)>,
    cregs: HashMap<Symbol, (u64, u64)>,
    params: HashMap<Symbol, Value>,
}

struct Definition {
    args: Vec<Symbol>,
    params: Vec<Symbol>,
    gates: Option<Vec<Span<Stmt>>>,
}

/// High-level translation interface.
///
/// This structure is used to turn a program into a linear
/// list of gates. It uses a value that implements `GateWriter`
/// to output primitive gates to some medium.
///
/// This can be used by constructing one with `Linearize::new`
/// with your chosen `GateWriter`, and then using the `ProgramVisitor`
/// impl to call `.visit_program` on your program. The `depth` argument
/// to `Linearize::new` determines how many layers of definitions
/// it will expand.
///
/// Note that it is assumed you have type-checked your program first.
/// If you haven't you may get garbage output / random panics. If
/// you have type-checked it, you shouldn't get any panics.
///
/// Example:
/// ```ignore
/// struct GatePrinter;
///
/// impl GateWriter for GatePrinter {
///     type Error = std::convert::Infallible;
/// 
///     fn initialize(&mut self, _: &[Symbol], _: &[Symbol]) -> Result<(), Self::Error> {
///         Ok(())
///     }
///
///     fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error> {
///         println!("cx {copy} {xor}");
///         Ok(())
///     }
///
///     fn write_u(&mut self, theta: Value, phi: Value, lambda: Value, reg: usize) -> Result<(), Self::Error> {
///         println!("u({theta}, {phi}, {lambda}) {reg}");
///         Ok(())
///     }
///
///     fn write_opaque(&mut self, name: &Symbol, _: &[Value], _: &[usize]) -> Result<(), Self::Error> {
///         println!("opaque gate {}", name);
///         Ok(())
///     }
///
///     fn write_barrier(&mut self, _: &[usize]) -> Result<(), Self::Error> {
///         Ok(())
///     }
///
///     fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error> {
///         println!("measure {} -> {}", from, to);
///         Ok(())
///     }
///
///     fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error> {
///         println!("reset {reg}");
///         Ok(())
///     }
///
///     fn start_conditional(&mut self, reg: usize, count: usize, value: usize) -> Result<(), Self::Error> {
///         println!("if ({reg}:{count} == {value}) {{");
///         Ok(())
///     }
///
///     fn end_conditional(&mut self) -> Result<(), Self::Error> {
///         println!("}}");
///         Ok(())
///     }
/// }
///
/// fn main() {
///     let program = ...; // acquire a program from somewhere.
///     program.type_check().unwrap(); // make sure to type check.
///     let mut l = Linearize::new(GatePrinter, usize::MAX);
///     l.visit_program(&program).unwrap();
/// }
/// ```
pub struct Linearize<T> {
    next_qid: u64,
    next_cid: u64,
    depth: usize,
    defs: HashMap<Symbol, Rc<Definition>>,
    stack: Vec<Frame>,
    frame: Frame,
    initialized: bool,
    writer: T,
}

/// Output format from `Linearize`.
///
/// This trait is used by `Linearize` to actually output the
/// linearized program. It contains various methods to output
/// a primitive operation. Qubits and bits are referenced by
/// consecutive integers starting from zero, and parameters
/// are provided as `Value`s.
///
/// The only non-obvious methods are `initialize`, which provides
/// the names of qubits and bits to the backend, and is called
/// exactly once, before any other function. `start/end_conditional`
/// is called before and after any statements that are intended
/// to be conditional on a classical value.
pub trait GateWriter: Sized {
    type Error: std::error::Error + 'static;

    fn initialize(&mut self, qubits: &[Symbol], bits: &[Symbol]) -> Result<(), Self::Error>;
    fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error>;
    fn write_u(&mut self, theta: Value, phi: Value, lambda: Value, reg: usize) -> Result<(), Self::Error>;
    fn write_opaque(&mut self, name: &Symbol, params: &[Value], args: &[usize]) -> Result<(), Self::Error>;
    fn write_barrier(&mut self, regs: &[usize]) -> Result<(), Self::Error>;
    fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error>;
    fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error>;
    fn start_conditional(&mut self, reg: usize, count: usize, val: u64) -> Result<(), Self::Error>;
    fn end_conditional(&mut self) -> Result<(), Self::Error>;
}

impl<T: GateWriter> Linearize<T> {
    pub fn new(writer: T, depth: usize) -> Linearize<T> {
        Linearize {
            next_qid: 0,
            next_cid: 0,
            depth,
            defs: HashMap::new(),
            stack: Vec::new(),
            frame: Frame {
                name: Symbol::new("<toplevel>"),
                def_name: Symbol::new("<toplevel>"),
                call: None,
                qregs: HashMap::new(),
                cregs: HashMap::new(),
                params: HashMap::new(),
            },
            initialized: false,
            writer,
        }
    }

    // Check that these args are distinct.
    fn assert_different(
        &mut self,
        a: usize,
        b: usize,
        aspan: FileSpan,
        bspan: FileSpan,
    ) -> Result<(), LinearizeError> {
        if a == b {
            self.err(Err(LinearizeErrorKind::OverlappingRegs {
                a: aspan,
                b: bspan,
            }))
        } else {
            Ok(())
        }
    }

    // Convert an error kind into an actual error
    // by recording a backtrace.
    fn err<V>(&self, e: Result<V, LinearizeErrorKind>) -> Result<V, LinearizeError> {
        e.map_err(|kind| {
            let mut stack: Vec<_> = self
                .stack
                .iter()
                .filter_map(|frame| Some((frame.name.clone(), frame.call?)))
                .collect();

            if let Some(call) = self.frame.call {
                stack.push((self.frame.name.clone(), call));
            }

            LinearizeError { stack, kind }
        })
    }

    // Convert a register reference to a qubit number and a size.
    // The qubit number corresponds to the base of this reference,
    // and the remaining qubits in the register are the next
    // consecutive size bits.
    fn resolve_qreg(&self, reg: &Reg) -> (u64, u64) {
        match reg.index {
            None => self.frame.qregs[&reg.name],
            Some(idx) => {
                let base = self.frame.qregs[&reg.name].0;
                (base + idx, 1)
            }
        }
    }

    fn resolve_creg(&self, reg: &Reg) -> (u64, u64) {
        match reg.index {
            None => self.frame.cregs[&reg.name],
            Some(idx) => {
                let base = self.frame.cregs[&reg.name].0;
                (base + idx, 1)
            }
        }
    }
}

impl<T: GateWriter> ProgramVisitor for Linearize<T> {
    type Error = LinearizeError;

    // Since ProgramVisitor guarantees that all the definitions
    // and register declarations are visited first, this is ok
    // to do at the same time as everything else.
    fn visit_opaque_def(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
    ) -> Result<(), Self::Error> {
        let def = Definition {
            args: args.iter().map(|s| s.to_symbol()).collect(),
            params: params.iter().map(|s| s.to_symbol()).collect(),
            gates: None,
        };

        self.defs.insert(name.to_symbol(), Rc::new(def));

        Ok(())
    }

    fn visit_gate_def(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
        gates: &[Span<Stmt>],
    ) -> Result<(), Self::Error> {
        let def = Definition {
            args: args.iter().map(|s| s.to_symbol()).collect(),
            params: params.iter().map(|s| s.to_symbol()).collect(),
            gates: Some(gates.to_vec()),
        };

        self.defs.insert(name.to_symbol(), Rc::new(def));

        Ok(())
    }

    fn visit_qreg(&mut self, reg: &Span<Reg>) -> Result<(), Self::Error> {
        // Allocate some registers to the current frame
        // and bump `next_qid` by the appropriate amount,
        // so that the qubits are layed out consecutively.
        let size = reg.index.unwrap_or(1);
        self.frame
            .qregs
            .insert(reg.name.to_symbol(), (self.next_qid, size));
        self.next_qid += size;

        Ok(())
    }

    fn visit_creg(&mut self, reg: &Span<Reg>) -> Result<(), Self::Error> {
        let size = reg.index.unwrap_or(1);
        self.frame
            .cregs
            .insert(reg.name.to_symbol(), (self.next_cid, size));
        self.next_cid += size;

        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &Span<Stmt>) -> Result<(), Self::Error> {
        // Since this is first called after all register declarations,
        // now is a good time to initialize the backend.
        if !self.initialized {
            let mut qubits = vec![Symbol::new(""); self.next_qid as usize];
            let mut bits = vec![Symbol::new(""); self.next_cid as usize];

            for (name, (base, size)) in &self.frame.qregs {
                for offset in 0..*size {
                    let n = Symbol::new(format!("{}[{}]", name, offset));
                    qubits[(*base + offset) as usize] = n;
                }
            }

            for (name, (base, size)) in &self.frame.cregs {
                for offset in 0..*size {
                    let n = Symbol::new(format!("{}[{}]", name, offset));
                    bits[(*base + offset) as usize] = n;
                }
            }

            let err = self.writer.initialize(&qubits, &bits).map_err(|error| {
                LinearizeErrorKind::WriterError {
                    error: Box::new(error)
                }
            });
            self.err(err)?;

            self.initialized = true;
        }

        self.walk_stmt(stmt)
    }

    fn visit_barrier(&mut self, regs: &[Span<Reg>]) -> Result<(), Self::Error> {
        // Get all regs referenced by these expressions
        // regardless of size of matching.
        let mut args = HashSet::new();
        for reg in regs {
            let (base, size) = self.resolve_qreg(reg);
            for offset in 0..size {
                args.insert((base + offset) as usize);
            }
        }

        let args = args.drain().collect::<Vec<_>>();
        let err = self.writer.write_barrier(&args).map_err(|error| {
            LinearizeErrorKind::WriterError {
                error: Box::new(error)
            }
        });
        self.err(err)?;

        Ok(())
    }

    fn visit_measure(&mut self, from: &Span<Reg>, to: &Span<Reg>) -> Result<(), Self::Error> {
        let (fbase, fsize) = self.resolve_qreg(from);
        let (tbase, tsize) = self.resolve_creg(to);

        if fsize == tsize {
            // Many - many
            for offset in 0..fsize {
                let err = self.writer
                    .write_measure((fbase + offset) as usize, (tbase + offset) as usize)
                    .map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                self.err(err)?;
            }
        } else if fsize == 1 {
            // One - many
            for offset in 0..tsize {
                let err = self.writer
                    .write_measure(fbase as usize, (tbase + offset) as usize)
                    .map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                self.err(err)?;
            }
        } else {
            // Many - one
            for offset in 0..fsize {
                let err = self.writer
                    .write_measure((fbase + offset) as usize, tbase as usize)
                    .map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                self.err(err)?;
            }
        }

        Ok(())
    }

    fn visit_reset(&mut self, reg: &Span<Reg>) -> Result<(), Self::Error> {
        let (base, size) = self.resolve_qreg(reg);
        for offset in 0..size {
            let err = self.writer.write_reset((base + offset) as usize)
            .map_err(|error| {
                LinearizeErrorKind::WriterError {
                    error: Box::new(error)
                }
            });
            self.err(err)?;
        }

        Ok(())
    }

    fn visit_cx(&mut self, copy: &Span<Reg>, xor: &Span<Reg>) -> Result<(), Self::Error> {
        let (cbase, csize) = self.resolve_qreg(copy);
        let (xbase, xsize) = self.resolve_qreg(xor);

        // Same as with visit_measure but this time they have to be
        // distinct as the CX of two identical gates is not well defined.
        if csize == xsize {
            for offset in 0..csize {
                self.assert_different(
                    (cbase + offset) as usize,
                    (xbase + offset) as usize,
                    copy.span,
                    xor.span,
                )?;
                let err = self.writer
                    .write_cx((cbase + offset) as usize, (xbase + offset) as usize)
                    .map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                self.err(err)?;
            }
        } else if csize == 1 {
            for offset in 0..xsize {
                self.assert_different(
                    cbase as usize,
                    (xbase + offset) as usize,
                    copy.span,
                    xor.span,
                )?;
                let err = self.writer
                    .write_cx(cbase as usize, (xbase + offset) as usize)
                    .map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                self.err(err)?;
            }
        } else {
            for offset in 0..csize {
                self.assert_different(
                    (cbase + offset) as usize,
                    xbase as usize,
                    copy.span,
                    xor.span,
                )?;
                let err = self.writer
                    .write_cx((cbase + offset) as usize, xbase as usize)
                    .map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                self.err(err)?;
            }
        }

        Ok(())
    }

    fn visit_u(
        &mut self,
        theta: &Span<Expr>,
        phi: &Span<Expr>,
        lambda: &Span<Expr>,
        reg: &Span<Reg>,
    ) -> Result<(), Self::Error> {
        // Evaluate the parameters in this frame.
        let mut eval = FrameEvaluator { frame: &self.frame };

        let theta = self.err(eval.visit_expr(theta).map_err(|e| match e {
            // Add the span information to the raw errors.
            EvalError::DivideByZero => LinearizeErrorKind::DivideByZero { span: theta.span },
            EvalError::ApproximateFail(value) => LinearizeErrorKind::ApproximateFail {
                span: theta.span,
                value,
            },
            EvalError::OverflowError => LinearizeErrorKind::NumericalOverflow { span: theta.span },
        }))?;
        let phi = self.err(eval.visit_expr(phi).map_err(|e| match e {
            EvalError::DivideByZero => LinearizeErrorKind::DivideByZero { span: phi.span },
            EvalError::ApproximateFail(value) => LinearizeErrorKind::ApproximateFail {
                span: phi.span,
                value,
            },
            EvalError::OverflowError => LinearizeErrorKind::NumericalOverflow { span: phi.span },
        }))?;
        let lambda = self.err(eval.visit_expr(lambda).map_err(|e| match e {
            EvalError::DivideByZero => LinearizeErrorKind::DivideByZero { span: lambda.span },
            EvalError::ApproximateFail(value) => LinearizeErrorKind::ApproximateFail {
                span: lambda.span,
                value,
            },
            EvalError::OverflowError => LinearizeErrorKind::NumericalOverflow { span: lambda.span },
        }))?;

        // Now do a unitary for each referenced qubit.
        let (base, size) = self.resolve_qreg(reg);
        for offset in 0..size {
            let err = self.writer
                .write_u(theta, phi, lambda, (base + offset) as usize)
                .map_err(|error| {
                    LinearizeErrorKind::WriterError {
                        error: Box::new(error)
                    }
                });
            self.err(err)?;
        }

        Ok(())
    }

    fn visit_gate(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Expr>],
        args: &[Span<Reg>],
    ) -> Result<(), Self::Error> {
        let def = self.defs[name.as_symbol()].clone();

        // We may need to push a new stack frame.
        let mut frame = Frame {
            name: self.frame.def_name.to_symbol(),
            def_name: name.to_symbol(),
            call: Some(name.span),
            cregs: HashMap::new(),
            qregs: HashMap::new(),
            params: HashMap::new(),
        };

        // First, evaluate all the parameters.
        let mut eval = FrameEvaluator { frame: &self.frame };

        let mut values = Vec::new();
        for (param, name) in params.iter().zip(&def.params) {
            let value = self.err(eval.visit_expr(param).map_err(|e| match e {
                EvalError::DivideByZero => LinearizeErrorKind::DivideByZero { span: param.span },
                EvalError::ApproximateFail(value) => LinearizeErrorKind::ApproximateFail {
                    span: param.span,
                    value,
                },
                EvalError::OverflowError => {
                    LinearizeErrorKind::NumericalOverflow { span: param.span }
                }
            }))?;

            // If this is opaque, we will need the parameters as a list.
            if def.gates.is_none() {
                values.push(value);
            } else {
                // Otherwise, just add them to the new stack frame.
                frame.params.insert(name.clone(), value);
            }
        }

        // Find the maximum register size. Since these are matched,
        // this is also the size of all non-single registers.
        let size = args
            .iter()
            .map(|a| self.resolve_qreg(a).1)
            .max()
            .unwrap_or(0);
        // Record which registers are fixed, and which are looped over.
        let fixed = args
            .iter()
            .map(|a| self.resolve_qreg(a).1 == 1)
            .collect::<Vec<_>>();
        // Record the base addresses of all registers.
        let mut argsn = args
            .iter()
            .map(|a| self.resolve_qreg(a).0 as usize)
            .collect::<Vec<_>>();

        if def.gates.is_some() {
            // Push the current frame onto the stack and replace it with the new one.
            self.stack.push(std::mem::replace(&mut self.frame, frame));
        }

        for _ in 0..size {
            // If we have an opaque gate, we must have all the qubits distinct.
            if def.gates.is_none() {
                for i in 0..argsn.len() {
                    for j in 0..i {
                        self.assert_different(argsn[i], argsn[j], args[i].span, args[j].span)?;
                    }
                }
            }

            match &def.gates {
                Some(gates) if self.stack.len() < self.depth => {
                    // This gate has a body, so process it. First insert
                    // all of the arguments as quantum registers of size one.
                    for (name, arg) in def.args.iter().zip(&argsn) {
                        self.frame.qregs.insert(name.clone(), (*arg as u64, 1));
                    }

                    // Then recurse on the body.
                    for stmt in gates {
                        self.visit_stmt(stmt)?;
                    }
                }
                _ => {
                    // If this is opaque (or we passed the depth limit), write it out straight away.
                    let err = self.writer.write_opaque(name, &values, &argsn).map_err(|error| {
                        LinearizeErrorKind::WriterError {
                            error: Box::new(error)
                        }
                    });
                    self.err(err)?;
                }
            }

            // Advance all the large registers in lock-step.
            for i in 0..argsn.len() {
                if !fixed[i] {
                    argsn[i] += 1;
                }
            }
        }

        // If we had a new stack frame, pop it off.
        if def.gates.is_some() {
            self.frame = self.stack.pop().unwrap();
        }

        Ok(())
    }

    fn visit_conditional(
        &mut self,
        reg: &Span<Reg>,
        val: &Span<u64>,
        then: &Span<Stmt>,
    ) -> Result<(), Self::Error> {
        let (base, size) = self.frame.qregs[&reg.name];
        let err =self.writer
            .start_conditional(base as usize, size as usize, **val).map_err(|error| {
                LinearizeErrorKind::WriterError {
                    error: Box::new(error)
                }
            });
        self.err(err)?;
        self.visit_stmt(then)?;
        let err = self.writer.end_conditional().map_err(|error| {
            LinearizeErrorKind::WriterError {
                error: Box::new(error)
            }
        });
        self.err(err)?;
        Ok(())
    }
}

/// An error that occured while linearizing a program.
/// This error contains both the call stack leading up
/// to this error, as well as the kind of error that occured.
/// `stack` is layed out as a list of function names and call
/// for those functions, from oldest to youngest (i.e the
/// current stack frame is the end of `stack`).
#[derive(Debug, Error)]
#[error("linearization error")]
pub struct LinearizeError {
    pub stack: Vec<(Symbol, FileSpan)>,
    #[source]
    pub kind: LinearizeErrorKind,
}

/// The type of error that occurred while linearizing.
#[derive(Debug, Error)]
pub enum LinearizeErrorKind {
    /// A division by zero happened while computing this parameter.
    #[error("division by zero")]
    DivideByZero { span: FileSpan },
    /// A value could not be converted from float to rational
    /// while computing this parameter.
    #[error("float approximation fail")]
    ApproximateFail { span: FileSpan, value: f32 },
    /// This expression had an overflow.
    #[error("numerical overflow")]
    NumericalOverflow { span: FileSpan },
    /// A `CX` gate or opaque gate was called with non-distinct arguments.
    #[error("overlapping arguments")]
    OverlappingRegs {
        /// This argument overlaps with the other.
        a: FileSpan,
        /// This argument overlaps with the other.
        b: FileSpan,
    },
    #[error("{error}")]
    WriterError {
        error: Box<dyn std::error::Error>
    }
}

#[cfg(feature = "ariadne")]
impl LinearizeError {
    pub fn to_report(&self) -> Report {
        let mut base = match self.kind {
            LinearizeErrorKind::DivideByZero { span } => {
                Report::build(ReportKind::Error, span.file, span.start)
                    .with_message("Division by zero in parameter evaluation")
                    .with_label(Label::new(span)
                        .with_message("This expression had an error.")
                        .with_color(ariadne::Color::Cyan))
            },
            LinearizeErrorKind::NumericalOverflow { span } => {
                Report::build(ReportKind::Error, span.file, span.start)
                    .with_message("Numerical overflow in parameter evaluation")
                    .with_label(Label::new(span)
                        .with_message("This expression had an error.")
                        .with_color(ariadne::Color::Cyan))
            },
            LinearizeErrorKind::ApproximateFail { value, span } => {
                Report::build(ReportKind::Error, span.file, span.start)
                    .with_message("Could not approximate float in parameter evaluation")
                    .with_label(Label::new(span)
                        .with_message(format!("Part of this expression evaluated to {}, which cannot be represented.", value))
                        .with_color(ariadne::Color::Cyan))
                    .with_note(concat!(
                        "Parameter values are represented in the form `a + bπ` for `a` and `b` rationals to preserve accuracy.",
                        "Not all floats can be represented in this form, however this usually indicates an error in your calculations as unrepresentable",
                        "values are not physically meaningful (they are either very large or invalid)."
                    ))
            },
            LinearizeErrorKind::OverlappingRegs { a, b } => {
                Report::build(ReportKind::Error, a.file, a.start)
                    .with_message("Overlapping arguments to primitive gate")
                    .with_label(Label::new(a)
                        .with_message("The register is first referenced here.")
                        .with_color(ariadne::Color::Cyan)
                        .with_order(0))
                    .with_label(Label::new(b)
                        .with_message("This overlaps with the previous reference.")
                        .with_color(ariadne::Color::Cyan)
                        .with_order(1))
            },
            LinearizeErrorKind::WriterError { ref error } => {
                Report::build(ReportKind::Error, 0usize, 0usize)
                    .with_message(format!("{}", error))
            }
        };

        let mut cols = ariadne::ColorGenerator::new();
        for (i, (name, call)) in self.stack.iter().enumerate().rev() {
            let kind = match i {
                0 => "Top level frame".to_string(),
                _ => format!("Frame {}", i),
            };

            let rest = if name.as_str() == "<toplevel>" {
                String::new()
            } else {
                format!(" in `{}`", name)
            };

            base = base.with_label(
                Label::new(*call)
                    .with_message(format!("{} originated here{}.", kind, rest))
                    .with_color(cols.next()),
            );
        }

        base.finish()
    }
}
