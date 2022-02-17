use crate::ast::{Decl, Expr, FileSpan, Program, Reg, Span, Stmt, Symbol};
use petgraph as px;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[cfg(feature = "ariadne")]
use {
    crate::ast::Report,
    ariadne::{Label, ReportKind},
};

impl Program {
    /// Type-check this whole program.
    ///
    /// This will check the program for some of the following errors:
    /// * Undefined gates, registers, or parameters,
    /// * Overlapping or invalid definitions or names,
    /// * Mismatched types and sizes in statements,
    /// * Recursive gate definitions.
    ///
    /// Notably, this does not check that arguments to gates
    /// are distinct, nor that parameters are valid (e.g free
    /// from divide-by-zero), as it does not do any evaluation.
    /// Therefore, some more errors may occur during interpretation,
    /// even if no errors are reported here.
    pub fn type_check(&self) -> Result<(), Vec<TypeError>> {
        let mut checker = TypeChecker {
            program: self,
            decls: HashMap::new(),
            errors: Vec::new(),
            refs: px::Graph::new(),
        };

        checker.check();

        if checker.errors.is_empty() {
            Ok(())
        } else {
            Err(checker.errors)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RegType {
    Classical(usize),
    Quantum(usize),
}

struct TypeChecker<'a> {
    program: &'a Program,
    decls: HashMap<Symbol, (px::graph::NodeIndex, usize, usize, FileSpan)>,
    errors: Vec<TypeError>,
    refs: px::Graph<Symbol, FileSpan>,
}

impl<'a> TypeChecker<'a> {
    fn check(&mut self) {
        // Get all the gate definitions beforehand,
        // so that they can be in any order.
        self.collect_decls();

        // Check that the body of the program is correct.
        self.check_gates();

        // Check that all the gate definitions are valid.
        self.check_defs();

        // Check that there are no recursive definitions.
        self.check_recursion();
    }

    fn err(&mut self, report: TypeError) {
        self.errors.push(report);
    }

    fn collect_decls(&mut self) {
        for decl in &self.program.decls {
            if let Decl::Def {
                name, params, args, ..
            } = &*decl.inner
            {
                if let Some((_, _, _, prev)) = self.decls.get(&*name.inner) {
                    let prev = *prev;
                    self.err(TypeError::RedefinedGate {
                        name: name.clone(),
                        decl: decl.span,
                        prev,
                    });
                } else {
                    self.decls.insert(
                        name.inner.to_symbol(),
                        (
                            self.refs.add_node(name.inner.to_symbol()),
                            params.len(),
                            args.len(),
                            decl.span,
                        ),
                    );

                    let mut set = HashMap::new();
                    for arg in args {
                        if let Some(prev) = set.insert(arg.inner.as_str(), arg.span) {
                            self.err(TypeError::RedefinedArgument {
                                name: arg.clone(),
                                decl: decl.span,
                                prev,
                            });
                        }
                    }

                    set.clear();
                    for arg in params {
                        if let Some(prev) = set.insert(arg.inner.as_str(), arg.span) {
                            self.err(TypeError::RedefinedParameter {
                                name: arg.clone(),
                                decl: decl.span,
                                prev,
                            });
                        }
                    }
                }
            }
        }
    }

    fn check_gates(&mut self) {
        // Since the body of the program has no arguments,
        // collect all register definitions as arguments.
        let regs = self.collect_regs();

        // Make a node in the call-graph to represent this.
        let node = self.refs.add_node(Symbol::new(""));

        let mut checker = FuncTypeChecker {
            decls: &self.decls,
            regs,
            // There are no parameters at the top-level
            params: HashSet::new(),
            errors: &mut self.errors,
            node,
            refs: &mut self.refs,
        };

        // Then just check it like any other definition.
        for decl in &self.program.decls {
            if let Decl::Stmt(stmt) = &*decl.inner {
                checker.check_stmt(stmt);
            }
        }
    }

    fn check_defs(&mut self) {
        for decl in &self.program.decls {
            if let Decl::Def {
                params,
                args,
                body,
                name,
            } = &*decl.inner
            {
                if let Some(body) = body.as_ref() {
                    let params = params.iter().map(|s| s.inner.to_symbol()).collect();
                    // Arguments to a gate are required to be qubit registers of size one
                    // by the specification:
                    let regs = args
                        .iter()
                        .map(|s| {
                            let name = s.inner.to_symbol();
                            let span = Span {
                                span: s.span,
                                inner: Box::new(RegType::Quantum(1)),
                            };
                            (name, span)
                        })
                        .collect();

                    // Get the node on the call-graph that correponds to this,
                    // and check it:
                    let (node, _, _, _) = self.decls.get(name.inner.as_symbol()).unwrap();
                    let mut checker = FuncTypeChecker {
                        decls: &self.decls,
                        regs,
                        params,
                        errors: &mut self.errors,
                        node: *node,
                        refs: &mut self.refs,
                    };

                    for stmt in body {
                        checker.check_stmt(stmt);
                    }
                }
            }
        }
    }

    fn check_recursion(&mut self) {
        // Here we will analyze the call-graph to find any
        // recursion. Recursion manifests as a cycle in the graph,
        // so to find and report it, we first split the graph into
        // strongly connected components.
        let scc = px::algo::tarjan_scc(&self.refs);
        // A strongly connected component represents a set of nodes
        // (i.e definitions) that are all reachable from each other
        // (i.e mutually recursive). We report one error per SCC
        // so as to avoid cluttering the output.
        for component in scc {
            match component.len() {
                // A component of size zero is invalid.
                0 => (),
                // A component of size one is either a valid definition,
                // or directly recursive if it has a self-edge.
                1 => {
                    if let Some(edge) = self.refs.find_edge(component[0], component[0]) {
                        // If there is a self-edge, we have direct recursion.
                        let (_, _, _, span) = self.decls[&self.refs[component[0]]];
                        let call = self.refs[edge];
                        self.err(TypeError::RecursiveDefinition {
                            cycle: vec![(span, call)],
                        });
                    }
                }
                // A component of size greater than one is a set of mutually
                // recursive definitions.
                _ => {
                    // Find a neighbour of component[0] that is in the component
                    let next = self
                        .refs
                        .neighbors(component[0])
                        .find(|idx| component.contains(idx))
                        .unwrap();
                    // Find a path from the neighbour back to the start node
                    // and thus we have a cycle starting at component[0].
                    let (_, path) = px::algo::astar(
                        &self.refs,
                        next,
                        |node| node == component[0],
                        |_| 1,
                        |_| 0,
                    )
                    .unwrap();

                    // Report an error based on this cycle. This will probably
                    // not be the only cycle in this SCC, and probably won't visit
                    // all the nodes in the component. The report therefore has a note
                    // to say that there may be other recursion cycles.
                    let mut prev = component[0];
                    let mut cycle = Vec::new();
                    for node in path {
                        let (_, _, _, span) = self.decls[&self.refs[node]];
                        let eidx = self.refs.find_edge(prev, node).unwrap();
                        let edge = self.refs[eidx];
                        prev = node;

                        cycle.push((span, edge));
                    }

                    self.err(TypeError::RecursiveDefinition { cycle });
                }
            }
        }
    }

    fn collect_regs(&mut self) -> HashMap<Symbol, Span<RegType>> {
        let mut regs = HashMap::new();
        for decl in &self.program.decls {
            match &*decl.inner {
                Decl::CReg { reg } | Decl::QReg { reg } => {
                    let size = reg.inner.index.unwrap_or(1);
                    if size == 0 {
                        self.err(TypeError::ZeroSizeRegister {
                            reg: reg.span,
                            decl: decl.span,
                        });
                    }

                    if let Some(prev) = regs.insert(
                        reg.inner.name.clone(),
                        Span {
                            span: reg.span,
                            inner: Box::new(match &*decl.inner {
                                Decl::CReg { .. } => RegType::Classical(size),
                                Decl::QReg { .. } => RegType::Quantum(size),
                                _ => unreachable!(),
                            }),
                        },
                    ) {
                        self.err(TypeError::RedefinedRegister {
                            reg: reg.clone(),
                            decl: decl.span,
                            prev: prev.span,
                        })
                    }
                }
                _ => (),
            }
        }
        regs
    }
}

struct FuncTypeChecker<'a> {
    decls: &'a HashMap<Symbol, (px::graph::NodeIndex, usize, usize, FileSpan)>,
    regs: HashMap<Symbol, Span<RegType>>,
    params: HashSet<Symbol>,
    errors: &'a mut Vec<TypeError>,
    node: px::graph::NodeIndex,
    refs: &'a mut px::Graph<Symbol, FileSpan>,
}

impl<'a> FuncTypeChecker<'a> {
    fn check_stmt(&mut self, stmt: &Span<Stmt>) {
        match &*stmt.inner {
            Stmt::U {
                theta,
                phi,
                lambda,
                reg,
            } => {
                // The argument to U is quantum.
                self.assert_reg(reg, stmt, false);
                // Its parameters must be valid.
                self.check_expr(theta, stmt);
                self.check_expr(phi, stmt);
                self.check_expr(lambda, stmt);
            }
            Stmt::CX { copy, xor } => {
                // Both arguments to CX are quantum.
                let a = self.assert_reg(copy, stmt, false);
                let b = self.assert_reg(xor, stmt, false);
                // Their sizes must match (or be one).
                self.assert_match([(a, copy), (b, xor)], stmt);
            }
            Stmt::Measure { from, to } => {
                // We have `measure quantum -> classical;`
                let a = self.assert_reg(from, stmt, false);
                let b = self.assert_reg(to, stmt, true);
                // Again, the sizes must match.
                self.assert_match([(a, from), (b, to)], stmt);
            }
            Stmt::Reset { reg } => {
                self.assert_reg(reg, stmt, false);
            }
            Stmt::Barrier { regs } => {
                // All arguments to barrier must be quantum:
                for arg in regs {
                    self.assert_reg(arg, stmt, false);
                }
            }
            Stmt::Conditional { reg, val, then } => {
                // The conditional register is classical.
                let size = self.assert_reg(reg, stmt, true);
                // The consequent statement must be valid.
                self.check_stmt(then);
                // The scalar we're comparing the register to can't be too big.
                self.assert_scalar_size(val, size, reg, stmt);
            }
            Stmt::Gate { name, params, args } => {
                // Check that this gate is defined:
                if let Some((node, parity, aarity, def)) = self.assert_def(name, stmt) {
                    // We must match the arity of both arguments and parameters:
                    self.assert_len(parity, params.len(), def, &*name.inner, stmt, "parameters");
                    self.assert_len(aarity, args.len(), def, &*name.inner, stmt, "arguments");

                    // If this succeeds, add an edge to the call-graph
                    // to indicate this definition calls the other.
                    self.refs.add_edge(self.node, node, stmt.span);
                }

                let mut sizes = Vec::new();
                for arg in args {
                    // All the arguments should be quantum registers:
                    sizes.push((self.assert_reg(arg, stmt, false), arg));
                }
                // All their sizes must match
                self.assert_match(sizes, stmt);

                for param in params {
                    self.check_expr(param, stmt);
                }
            }
        }
    }

    fn check_expr(&mut self, expr: &Span<Expr>, stmt: &Span<Stmt>) {
        match &*expr.inner {
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                self.check_expr(a, stmt);
                self.check_expr(b, stmt);
            }

            Expr::Neg(a)
            | Expr::Ln(a)
            | Expr::Exp(a)
            | Expr::Sqrt(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tan(a) => self.check_expr(a, stmt),

            Expr::Var(s) => {
                if !self.params.contains(s) {
                    self.err::<()>(TypeError::UndefinedParameter {
                        name: s.clone(),
                        stmt: stmt.span,
                        span: expr.span,
                    });
                }
            }

            Expr::Int(_) | Expr::Real(_) | Expr::Pi => (),
        }
    }

    fn err<T>(&mut self, report: TypeError) -> Option<T> {
        self.errors.push(report);
        None
    }

    fn assert_scalar_size(
        &mut self,
        val: &Span<usize>,
        size: Option<usize>,
        reg: &Span<Reg>,
        stmt: &Span<Stmt>,
    ) {
        if let Some(size) = size {
            if *val.inner >= (1 << size) {
                // Get the size of the comparison value.
                // What we really want is like val.log2() + 1,
                // but integer log is unstable at the minute.
                let bsize =
                    val.inner
                        .checked_next_power_of_two()
                        .map_or(usize::BITS, |v| v.trailing_zeros()) as usize;
                self.err::<()>(TypeError::InvalidComparisonSize {
                    reg: reg.span,
                    value: val.span,
                    stmt: stmt.span,
                    reg_size: size,
                    value_size: bsize,
                });
            }
        }
    }

    fn assert_len(
        &mut self,
        arity: usize,
        args: usize,
        def: FileSpan,
        name: &Symbol,
        stmt: &Span<Stmt>,
        kind: &str,
    ) {
        if arity != args {
            if kind == "arguments" {
                self.err::<()>(TypeError::WrongArgumentArity {
                    stmt: stmt.span,
                    def,
                    actual: args,
                    arity,
                    name: name.clone(),
                });
            } else if kind == "parameters" {
                self.err::<()>(TypeError::WrongParameterArity {
                    stmt: stmt.span,
                    def,
                    actual: args,
                    arity,
                    name: name.clone(),
                });
            }
        }
    }

    fn assert_def(
        &mut self,
        name: &Span<Symbol>,
        stmt: &Span<Stmt>,
    ) -> Option<(px::graph::NodeIndex, usize, usize, FileSpan)> {
        match self.decls.get(&*name.inner) {
            Some(s) => Some(*s),
            None => self.err(TypeError::UndefinedGate {
                name: name.clone(),
                stmt: stmt.span,
            }),
        }
    }

    fn assert_match<'b, I>(&mut self, iter: I, stmt: &Span<Stmt>)
    where
        I: IntoIterator<Item = (Option<usize>, &'b Span<Reg>)>,
    {
        // The size and span corresponding to the
        // previously matched argument.
        let mut match_size = 1;
        let mut match_span = FileSpan::empty();
        for (size, reg) in iter {
            // Some arguments may not have sizes if there is an
            // error in their definition. In this case we ignore them
            // for matching.
            if let Some(size) = size {
                let span = reg.span;
                // If this is the first large register, we match against it.
                if match_size == 1 {
                    match_size = size;
                    match_span = span;
                }

                // If the size of the register is bigger than one, it
                // must match the others.
                if size > 1 && size != match_size {
                    self.err::<()>(TypeError::WrongOperandSize {
                        stmt: stmt.span,
                        span,
                        size,
                        match_span,
                        match_size,
                    });
                }
            }
        }
    }

    fn assert_reg(&mut self, reg: &Span<Reg>, stmt: &Span<Stmt>, classical: bool) -> Option<usize> {
        // Get the size of a register:
        let map_qubit = |rty: RegType| {
            if classical {
                match rty {
                    RegType::Classical(size) => Some(size),
                    _ => None,
                }
            } else {
                match rty {
                    RegType::Quantum(size) => Some(size),
                    _ => None,
                }
            }
        };

        // Check that the register is defined:
        match self.regs.get(&reg.inner.name) {
            Some(def @ Span { inner, .. }) => match map_qubit(**inner) {
                // Check that it is of the right type:
                Some(size) => {
                    if let Some(index) = reg.inner.index {
                        // Check that the index is in range:
                        if index < size {
                            Some(1)
                        } else {
                            let defspan = def.span;
                            self.err(TypeError::InvalidRegisterIndex {
                                stmt: stmt.span,
                                def: defspan,
                                reg: reg.span,
                                size,
                                index,
                                name: reg.inner.name.clone(),
                            })
                        }
                    } else {
                        Some(size)
                    }
                }
                None => {
                    let defspan = def.span;
                    self.err(TypeError::WrongRegisterType {
                        classical,
                        def: defspan,
                        stmt: stmt.span,
                        reg: reg.span,
                        name: reg.inner.name.clone(),
                    })
                }
            },
            None => self.err(TypeError::UndefinedRegister {
                stmt: stmt.span,
                reg: reg.span,
                name: reg.inner.name.clone(),
            }),
        }
    }
}

/// An error produced during type-checking.
///
/// This includes several main categories:
/// * `Redefined...`: two objects have the same name,
/// * `Wrong...`: the arity/type/size doesn't match what is required,
/// * `Undefined...`: the thing referred to doesn't exist,
/// * and a few other miscellaneous errors.
///
/// In general, `name` refers to the name of the thing that is
/// redefined/undefined/wrong, while `def`/`decl` refers to the place
/// it is defined, `stmt` refers to the statement where this error
/// occured, and `prev` is where something was previously defined or
/// referenced. `reg`/`span` refers to the object in question.
///
/// Documentation is given for fields that are not listed here.
///
#[derive(Debug, Error)]
pub enum TypeError {
    /// This gate has already been defined.
    #[error("multiple definition of gate")]
    RedefinedGate {
        name: Span<Symbol>,
        decl: FileSpan,
        prev: FileSpan,
    },
    /// This argument name has already been used.
    #[error("multiple definition of argument")]
    RedefinedArgument {
        name: Span<Symbol>,
        decl: FileSpan,
        prev: FileSpan,
    },
    /// This parameter name has already been used.
    #[error("multiple definition of parameter")]
    RedefinedParameter {
        name: Span<Symbol>,
        decl: FileSpan,
        prev: FileSpan,
    },
    /// This sequence of definitions is recursive.
    #[error("recursive definitions")]
    RecursiveDefinition {
        /// These definitions form a cycle.
        /// The first component refers to the definition,
        /// the second component to where it calls the next
        /// definition in the chain.
        cycle: Vec<(FileSpan, FileSpan)>,
    },
    /// A register has been declared with size zero.
    #[error("zero-size register declaration")]
    ZeroSizeRegister { reg: FileSpan, decl: FileSpan },
    /// This register has already been declared.
    #[error("multiple definition of register")]
    RedefinedRegister {
        reg: Span<Reg>,
        decl: FileSpan,
        prev: FileSpan,
    },
    /// This parameter is undefined.
    #[error("undefined parameter")]
    UndefinedParameter {
        name: Symbol,
        stmt: FileSpan,
        span: FileSpan,
    },
    /// This comparison is out of range.
    #[error("invalid comparison size")]
    InvalidComparisonSize {
        /// The classical register being compared.
        reg: FileSpan,
        /// The constant value it is compared to.
        value: FileSpan,
        stmt: FileSpan,
        /// The size of the register.
        reg_size: usize,
        /// The size of the constant value.
        value_size: usize,
    },
    /// This statement has the wrong number of arguments.
    #[error("incorrect argument arity")]
    WrongArgumentArity {
        stmt: FileSpan,
        def: FileSpan,
        name: Symbol,
        /// The correct number of arguments.
        arity: usize,
        /// The actual number of arguments.
        actual: usize,
    },
    /// This statement has the wrong number of parameters.
    #[error("incorrect parameter arity")]
    WrongParameterArity {
        stmt: FileSpan,
        def: FileSpan,
        name: Symbol,
        /// The correct number of parameters.
        arity: usize,
        /// The actual number of parameters.
        actual: usize,
    },
    /// This gate is undefined.
    #[error("undefined gate")]
    UndefinedGate { name: Span<Symbol>, stmt: FileSpan },
    /// This operand's size doesn't match the others in this statement.
    #[error("mismatched operand sizes")]
    WrongOperandSize {
        stmt: FileSpan,
        /// Reference to this operand.
        span: FileSpan,
        /// The size of this operand.
        size: usize,
        /// Reference to the operand it needs to match.
        match_span: FileSpan,
        /// The size of the operand it should match.
        match_size: usize,
    },
    /// This index is invalid for this register.
    #[error("invalid register index")]
    InvalidRegisterIndex {
        name: Symbol,
        /// The index being accessed.
        index: usize,
        /// The size of the register.
        size: usize,
        reg: FileSpan,
        def: FileSpan,
        stmt: FileSpan,
    },
    /// This operand has the wrong type.
    #[error("mismatched operand type")]
    WrongRegisterType {
        name: Symbol,
        def: FileSpan,
        stmt: FileSpan,
        reg: FileSpan,
        /// Whether or not the operand is supposed to be classical.
        classical: bool,
    },
    /// This register is undefined.
    #[error("undefined register")]
    UndefinedRegister {
        name: Symbol,
        reg: FileSpan,
        stmt: FileSpan,
    },
}

#[cfg(feature = "ariadne")]
impl TypeError {
    /// Convert this error into a `Report` for printing.
    pub fn to_report(&self) -> Report {
        match self {
            TypeError::RedefinedGate { name, decl, prev } => {
                Report::build(ReportKind::Error, decl.file, decl.start)
                    .with_message(format!("Multiple definition of `{}`", name.inner))
                    .with_label(
                        Label::new(name.span)
                            .with_message(format!("`{}` is redefined here.", name.inner))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .with_label(
                        Label::new(*prev)
                            .with_message(format!("`{}` was previously defined here.", name.inner))
                            .with_color(ariadne::Color::Green),
                    )
                    .finish()
            }
            TypeError::RedefinedArgument { name, decl, prev } => {
                Report::build(ReportKind::Error, decl.file, decl.start)
                    .with_message("Repeated argument name")
                    .with_label(
                        Label::new(name.span)
                            .with_message(format!("`{}` is used here,", name.inner))
                            .with_color(ariadne::Color::Cyan)
                            .with_order(0),
                    )
                    .with_label(
                        Label::new(*prev)
                            .with_message("but it was previously used here.")
                            .with_color(ariadne::Color::Green)
                            .with_order(1),
                    )
                    .finish()
            }
            TypeError::RedefinedParameter { name, decl, prev } => {
                Report::build(ReportKind::Error, decl.file, decl.start)
                    .with_message("Repeated parameter name")
                    .with_label(
                        Label::new(name.span)
                            .with_message(format!("`{}` is used here,", name.inner))
                            .with_color(ariadne::Color::Cyan)
                            .with_order(0),
                    )
                    .with_label(
                        Label::new(*prev)
                            .with_message("but it was previously used here.")
                            .with_color(ariadne::Color::Green)
                            .with_order(1),
                    )
                    .finish()
            }
            TypeError::RecursiveDefinition { cycle } => {
                if cycle.len() == 1 {
                    let (span, call) = cycle[0];
                    Report::build(ReportKind::Error, span.file, span.start)
                        .with_message("Recursive gate definition")
                        .with_label(
                            Label::new(span)
                                .with_message("This definition is recursive.")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .with_label(
                            Label::new(call)
                                .with_message("Recursion occurs here.")
                                .with_color(ariadne::Color::Green),
                        )
                        .with_note("Recursive definitions are not permitted.")
                        .finish()
                } else {
                    let span = cycle[0].0;
                    let mut builder = Report::build(ReportKind::Error, span.file, span.start)
                        .with_message("Mutually recursive gate definitions")
                        .with_note("The sequence of statements shown is not necessarily the only recursive cycle.");

                    for (span, call) in cycle {
                        builder = builder
                            .with_label(
                                Label::new(*span)
                                    .with_message("This definition is part of a cycle.")
                                    .with_color(ariadne::Color::Cyan),
                            )
                            .with_label(
                                Label::new(*call)
                                    .with_message("Recursion occurs here.")
                                    .with_color(ariadne::Color::Green),
                            );
                    }

                    builder.finish()
                }
            }
            TypeError::ZeroSizeRegister { reg, decl } => {
                Report::build(ReportKind::Error, decl.file, decl.start)
                    .with_message("Register declared with size zero")
                    .with_label(
                        Label::new(*reg)
                            .with_message("This register is declared with size zero.")
                            .with_color(ariadne::Color::Cyan),
                    )
                    .with_note("Register declarations must have positive size.")
                    .finish()
            }
            TypeError::RedefinedRegister { reg, decl, prev } => {
                Report::build(ReportKind::Error, decl.file, decl.start)
                    .with_message(format!(
                        "Multiple declaration of register `{}`",
                        reg.inner.name
                    ))
                    .with_label(
                        Label::new(*prev)
                            .with_message(format!(
                                "`{}` was previously declared here.",
                                reg.inner.name
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .with_label(
                        Label::new(reg.span)
                            .with_message(format!("`{}` is declared again here.", reg.inner.name))
                            .with_color(ariadne::Color::Green),
                    )
                    .finish()
            }
            TypeError::UndefinedParameter { name, stmt, span } => {
                Report::build(ReportKind::Error, stmt.file, stmt.start)
                    .with_message(format!("Undefined parameter `{}`", name))
                    .with_label(
                        Label::new(*span)
                            .with_message(format!(
                                "`{}` was referenced, but it is not defined.",
                                name
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish()
            }
            TypeError::InvalidComparisonSize {
                reg,
                value,
                stmt,
                reg_size,
                value_size,
            } => Report::build(ReportKind::Error, stmt.file, stmt.start)
                .with_message("Comparison value is too large")
                .with_label(
                    Label::new(*reg)
                        .with_message(format!("This register has size {}", reg_size))
                        .with_color(ariadne::Color::Cyan),
                )
                .with_label(
                    Label::new(*value)
                        .with_message(format!(
                            "but it is compared to this value of size {}.",
                            value_size
                        ))
                        .with_color(ariadne::Color::Green),
                )
                .finish(),
            TypeError::WrongArgumentArity {
                stmt,
                def,
                name,
                arity,
                actual,
            } => Report::build(ReportKind::Error, stmt.file, stmt.start)
                .with_message("Wrong number of arguments for gate")
                .with_label(
                    Label::new(*stmt)
                        .with_message(format!(
                            "In this statement {} arguments are provided.",
                            actual
                        ))
                        .with_color(ariadne::Color::Cyan),
                )
                .with_label(
                    Label::new(*def)
                        .with_message(format!(
                            "`{}` is defined here with {} arguments.",
                            name, arity
                        ))
                        .with_color(ariadne::Color::Green),
                )
                .finish(),
            TypeError::WrongParameterArity {
                stmt,
                def,
                name,
                arity,
                actual,
            } => Report::build(ReportKind::Error, stmt.file, stmt.start)
                .with_message("Wrong number of parameters for gate")
                .with_label(
                    Label::new(*stmt)
                        .with_message(format!(
                            "In this statement {} parameters are provided.",
                            actual
                        ))
                        .with_color(ariadne::Color::Cyan),
                )
                .with_label(
                    Label::new(*def)
                        .with_message(format!(
                            "`{}` is defined here with {} parameters.",
                            name, arity
                        ))
                        .with_color(ariadne::Color::Green),
                )
                .finish(),
            TypeError::UndefinedGate { stmt, name } => {
                Report::build(ReportKind::Error, stmt.file, stmt.start)
                    .with_message(format!("Undefined gate `{}`", name.inner))
                    .with_label(
                        Label::new(name.span)
                            .with_message(format!(
                                "`{}` was referenced, but it is not defined.",
                                name.inner
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish()
            }
            TypeError::WrongOperandSize {
                stmt,
                span,
                size,
                match_span,
                match_size,
            } => Report::build(ReportKind::Error, stmt.file, stmt.start)
                .with_message("Mismatched operand sizes")
                .with_label(
                    Label::new(*stmt)
                        .with_message("In this statement arguments must be of size one")
                        .with_color(ariadne::Color::Cyan),
                )
                .with_label(
                    Label::new(*match_span)
                        .with_message(format!("or of size {} to match this argument,", match_size))
                        .with_color(ariadne::Color::Green),
                )
                .with_label(
                    Label::new(*span)
                        .with_message(format!("but this argument has size {}.", size))
                        .with_color(ariadne::Color::Magenta),
                )
                .finish(),
            TypeError::InvalidRegisterIndex {
                name,
                index,
                size,
                reg,
                def,
                stmt,
            } => Report::build(ReportKind::Error, stmt.file, stmt.start)
                .with_message(format!("Register index `{}[{}]` out of range", name, index))
                .with_label(
                    Label::new(*def)
                        .with_message(format!("`{}` is defined here with size {}.", name, size))
                        .with_color(ariadne::Color::Cyan),
                )
                .with_label(
                    Label::new(*reg)
                        .with_message(format!("Index {} is referenced here.", index))
                        .with_color(ariadne::Color::Green),
                )
                .finish(),
            TypeError::WrongRegisterType {
                def,
                stmt,
                reg,
                name,
                classical,
            } => {
                let aname = if *classical { "quantum" } else { "classical" };
                let bname = if *classical { "classical" } else { "quantum" };
                Report::build(ReportKind::Error, stmt.file, stmt.start)
                    .with_message(format!("Mismatched types, expected {} register", bname))
                    .with_label(
                        Label::new(*def)
                            .with_message(format!("`{}` is defined here as {}.", name, aname))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .with_label(
                        Label::new(*reg)
                            .with_message(format!("A {} register was expected here.", bname))
                            .with_color(ariadne::Color::Green),
                    )
                    .finish()
            }
            TypeError::UndefinedRegister { stmt, reg, name } => {
                Report::build(ReportKind::Error, stmt.file, stmt.start)
                    .with_message(format!("Undefined register `{}`", name))
                    .with_label(
                        Label::new(*reg)
                            .with_message(format!(
                                "`{}` was referenced, but it is not defined.",
                                name
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish()
            }
        }
    }
}
