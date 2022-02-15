use crate::ast::{Decl, Expr, Program, Reg, Span, Stmt, Symbol};

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
///
/// For example, to get the names of all gate definitions in a program,
/// you might do the following:
/// ```rust
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
    fn visit_program(&mut self, program: &Span<Program>) {
        self.walk_program(program);
    }

    fn walk_program(&mut self, program: &Span<Program>) {
        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::Include { .. }) {
                self.visit_decl(decl);
            }
        }

        for decl in &program.decls {
            let decl = decl;
            if matches!(&*decl.inner, Decl::Def { .. }) {
                self.visit_decl(decl);
            }
        }

        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::QReg { .. }) {
                self.visit_decl(decl);
            }
        }

        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::CReg { .. }) {
                self.visit_decl(decl);
            }
        }

        for decl in &program.decls {
            if matches!(&*decl.inner, Decl::Stmt(..)) {
                self.visit_decl(decl);
            }
        }
    }

    fn visit_decl(&mut self, decl: &Span<Decl>) {
        self.walk_decl(decl)
    }

    fn walk_decl(&mut self, decl: &Span<Decl>) {
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

    fn visit_include(&mut self, file: &Span<Symbol>) {}

    fn visit_qreg(&mut self, reg: &Span<Reg>) {}

    fn visit_creg(&mut self, reg: &Span<Reg>) {}

    fn visit_opaque_def(
        &mut self,
        name: &Span<Symbol>,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
    ) {
    }

    fn visit_gate_def(
        &mut self,
        name: &Symbol,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
        body: &[Span<Stmt>],
    ) {
        self.walk_gate_def(name, params, args, body);
    }

    fn walk_gate_def(
        &mut self,
        name: &Symbol,
        params: &[Span<Symbol>],
        args: &[Span<Symbol>],
        body: &[Span<Stmt>],
    ) {
        for stmt in body {
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, stmt: &Span<Stmt>) {
        self.walk_stmt(stmt);
    }

    fn walk_stmt(&mut self, stmt: &Span<Stmt>) {
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

    fn visit_barrier(&mut self, regs: &[Span<Reg>]) {}

    fn visit_measure(&mut self, from: &Span<Reg>, to: &Span<Reg>) {}

    fn visit_reset(&mut self, reg: &Span<Reg>) {}

    fn visit_cx(&mut self, copy: &Span<Reg>, xor: &Span<Reg>) {}

    fn visit_u(
        &mut self,
        theta: &Span<Expr>,
        phi: &Span<Expr>,
        lambda: &Span<Expr>,
        reg: &Span<Reg>,
    ) {
    }

    fn visit_gate(&mut self, name: &Span<Symbol>, params: &[Span<Expr>], args: &[Span<Reg>]) {
        self.walk_gate(name, params, args);
    }

    fn walk_gate(&mut self, name: &Span<Symbol>, params: &[Span<Expr>], args: &[Span<Reg>]) {}

    fn visit_conditional(&mut self, reg: &Span<Reg>, val: &Span<usize>, then: &Span<Stmt>) {
        self.walk_conditional(reg, val, then);
    }

    fn walk_conditional(&mut self, reg: &Span<Reg>, val: &Span<usize>, then: &Span<Stmt>) {
        self.visit_stmt(then);
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
/// ```rust
/// struct MaxFinder;
///
/// impl ExprVisitor for MaxFinder {
///     type Output = usize;
///
///     fn binop(&mut self, _: Binop, a: usize, b: usize) -> usize { a.max(b) }
///     fn unop(&mut self, _: Unop, a: usize) -> usize { a }
///     fn lookup(&mut self, _: &Symbol) -> usize { 0 }
///     fn pi(&mut self) -> usize { 0 }
///     fn int(&mut self, val: usize) -> usize { val }
///     fn real(&mut self, val: f32) -> usize { 0 }
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
    fn int(&mut self, val: usize) -> Self::Output;
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
