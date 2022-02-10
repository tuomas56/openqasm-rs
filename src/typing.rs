use crate::ast::{Decl, Expr, FileSpan, Program, Reg, Report, Span, Stmt, Symbol};
use ariadne::{Label, ReportKind};
use petgraph as px;
use std::collections::{HashMap, HashSet};

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
    pub fn type_check(&self) -> Result<(), Vec<Report>> {
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
    errors: Vec<Report>,
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

    fn err(&mut self, report: Report) {
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
                    self.err(
                        Report::build(ReportKind::Error, decl.span.file, decl.span.start)
                            .with_message(format!("Multiple definition of `{}`", name.inner))
                            .with_label(
                                Label::new(name.span)
                                    .with_message(format!("`{}` is redefined here.", name.inner))
                                    .with_color(ariadne::Color::Cyan),
                            )
                            .with_label(
                                Label::new(prev)
                                    .with_message(format!(
                                        "`{}` was previously defined here.",
                                        name.inner
                                    ))
                                    .with_color(ariadne::Color::Green),
                            )
                            .finish(),
                    );
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
                            self.err(
                                Report::build(ReportKind::Error, decl.span.file, decl.span.start)
                                    .with_message("Repeated argument name")
                                    .with_label(
                                        Label::new(arg.span)
                                            .with_message(format!("`{}` is used here,", arg.inner))
                                            .with_color(ariadne::Color::Cyan)
                                            .with_order(0),
                                    )
                                    .with_label(
                                        Label::new(prev)
                                            .with_message("but it was previously used here.")
                                            .with_color(ariadne::Color::Green)
                                            .with_order(1),
                                    )
                                    .finish(),
                            );
                        }
                    }

                    set.clear();
                    for arg in params {
                        if let Some(prev) = set.insert(arg.inner.as_str(), arg.span) {
                            self.err(
                                Report::build(ReportKind::Error, decl.span.file, decl.span.start)
                                    .with_message("Repeated parameter name")
                                    .with_label(
                                        Label::new(arg.span)
                                            .with_message(format!("`{}` is used here,", arg.inner))
                                            .with_color(ariadne::Color::Cyan)
                                            .with_order(0),
                                    )
                                    .with_label(
                                        Label::new(prev)
                                            .with_message("but it was previously used here.")
                                            .with_color(ariadne::Color::Green)
                                            .with_order(1),
                                    )
                                    .finish(),
                            );
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
                        self.err(
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
                                .finish(),
                        );
                    }
                }
                // A component of size greater than one is a set of mutually
                // recursive definitions.
                _ => {
                    // Get a span for the first component just for reference.
                    let (_, _, _, span) = self.decls[&self.refs[component[0]]];
                    let mut builder = Report::build(ReportKind::Error, span.file, span.start)
                        .with_message("Mutually recursive gate definitions")
                        .with_note("The sequence of statements shown is not necessarily the only recursive cycle.");

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
                    for node in path {
                        let (_, _, _, span) = self.decls[&self.refs[node]];
                        let eidx = self.refs.find_edge(prev, node).unwrap();
                        let edge = self.refs[eidx];
                        prev = node;

                        builder = builder
                            .with_label(
                                Label::new(span)
                                    .with_message("This definition is part of a cycle.")
                                    .with_color(ariadne::Color::Cyan),
                            )
                            .with_label(
                                Label::new(edge)
                                    .with_message("Recursion occurs here.")
                                    .with_color(ariadne::Color::Green),
                            );
                    }

                    self.err(builder.finish());
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
                        self.err(
                            Report::build(ReportKind::Error, decl.span.file, decl.span.start)
                                .with_message("Register declared with size zero")
                                .with_label(
                                    Label::new(reg.span)
                                        .with_message("This register is declared with size zero.")
                                        .with_color(ariadne::Color::Cyan),
                                )
                                .with_note("Register declarations must have positive size.")
                                .finish(),
                        )
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
                        self.err(
                            Report::build(ReportKind::Error, decl.span.file, decl.span.start)
                                .with_message(format!(
                                    "Multiple declaration of register `{}`",
                                    reg.inner.name
                                ))
                                .with_label(
                                    Label::new(prev.span)
                                        .with_message(format!(
                                            "`{}` was previously declared here.",
                                            reg.inner.name
                                        ))
                                        .with_color(ariadne::Color::Cyan),
                                )
                                .with_label(
                                    Label::new(reg.span)
                                        .with_message(format!(
                                            "`{}` is declared again here.",
                                            reg.inner.name
                                        ))
                                        .with_color(ariadne::Color::Green),
                                )
                                .finish(),
                        )
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
    errors: &'a mut Vec<Report>,
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
                    self.assert_len(
                        parity,
                        params.len(),
                        def,
                        &*name.inner,
                        stmt,
                        if params.len() == 1 {
                            "parameter"
                        } else {
                            "parameters"
                        },
                    );
                    self.assert_len(
                        aarity,
                        args.len(),
                        def,
                        &*name.inner,
                        stmt,
                        if args.len() == 1 {
                            "argument"
                        } else {
                            "arguments"
                        },
                    );

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

            Expr::Ln(a)
            | Expr::Exp(a)
            | Expr::Sqrt(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tan(a) => self.check_expr(a, stmt),

            Expr::Var(s) => {
                if !self.params.contains(s) {
                    self.err::<()>(
                        Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                            .with_message(format!("Undefined parameter `{}`", s))
                            .with_label(
                                Label::new(expr.span)
                                    .with_message(format!(
                                        "`{}` was referenced, but it is not defined.",
                                        s
                                    ))
                                    .with_color(ariadne::Color::Cyan),
                            )
                            .finish(),
                    );
                }
            }

            Expr::Int(_) | Expr::Real(_) | Expr::Pi => (),
        }
    }

    fn err<T>(&mut self, report: Report) -> Option<T> {
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
                let bsize = val
                    .inner
                    .checked_next_power_of_two()
                    .map_or(usize::BITS, |v| v.trailing_zeros());
                self.err::<()>(
                    Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                        .with_message("Comparison value is too large")
                        .with_label(
                            Label::new(reg.span)
                                .with_message(format!("This register has size {}", size))
                                .with_color(ariadne::Color::Cyan),
                        )
                        .with_label(
                            Label::new(val.span)
                                .with_message(format!(
                                    "but it is compared to this value of size {}.",
                                    bsize
                                ))
                                .with_color(ariadne::Color::Green),
                        )
                        .finish(),
                );
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
            self.err::<()>(
                Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                    .with_message(format!("Wrong number of {} for gate", kind))
                    .with_label(
                        Label::new(stmt.span)
                            .with_message(format!(
                                "In this statement {} {} are provided.",
                                args, kind
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .with_label(
                        Label::new(def)
                            .with_message(format!(
                                "`{}` is defined here with {} {}.",
                                name, arity, kind
                            ))
                            .with_color(ariadne::Color::Green),
                    )
                    .finish(),
            );
        }
    }

    fn assert_def(
        &mut self,
        name: &Span<Symbol>,
        stmt: &Span<Stmt>,
    ) -> Option<(px::graph::NodeIndex, usize, usize, FileSpan)> {
        match self.decls.get(&*name.inner) {
            Some(s) => Some(*s),
            None => self.err(
                Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                    .with_message(format!("Undefined gate `{}`", name.inner))
                    .with_label(
                        Label::new(name.span)
                            .with_message(format!(
                                "`{}` was referenced, but it is not defined.",
                                name.inner
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish(),
            ),
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
                    self.err::<()>(
                        Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                            .with_message("Mismatched operand sizes")
                            .with_label(
                                Label::new(stmt.span)
                                    .with_message("In this statement arguments must be of size one")
                                    .with_color(ariadne::Color::Cyan),
                            )
                            .with_label(
                                Label::new(match_span)
                                    .with_message(format!(
                                        "or of size {} to match this argument,",
                                        match_size
                                    ))
                                    .with_color(ariadne::Color::Green),
                            )
                            .with_label(
                                Label::new(span)
                                    .with_message(format!("but this argument has size {}.", size)),
                            )
                            .finish(),
                    );
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

        let aname = if classical { "quantum" } else { "classical" };
        let bname = if classical { "classical" } else { "quantum" };

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
                            self.err(
                                Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                                    .with_message(format!(
                                        "Register index `{}[{}]` out of range",
                                        reg.inner.name, index
                                    ))
                                    .with_label(
                                        Label::new(defspan)
                                            .with_message(format!(
                                                "`{}` is defined here with size {}.",
                                                reg.inner.name, size
                                            ))
                                            .with_color(ariadne::Color::Cyan),
                                    )
                                    .with_label(
                                        Label::new(reg.span)
                                            .with_message(format!(
                                                "Index {} is referenced here.",
                                                index
                                            ))
                                            .with_color(ariadne::Color::Green),
                                    )
                                    .finish(),
                            )
                        }
                    } else {
                        Some(size)
                    }
                }
                None => {
                    let defspan = def.span;
                    self.err(
                        Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                            .with_message(format!("Mismatched types, expected {} register", bname))
                            .with_label(
                                Label::new(defspan)
                                    .with_message(format!(
                                        "`{}` is defined here as {}.",
                                        reg.inner.name, aname
                                    ))
                                    .with_color(ariadne::Color::Cyan),
                            )
                            .with_label(
                                Label::new(reg.span)
                                    .with_message(format!(
                                        "A {} register was expected here.",
                                        bname
                                    ))
                                    .with_color(ariadne::Color::Green),
                            )
                            .finish(),
                    )
                }
            },
            None => self.err(
                Report::build(ReportKind::Error, stmt.span.file, stmt.span.start)
                    .with_message(format!("Undefined register `{}`", reg.inner.name))
                    .with_label(
                        Label::new(reg.span)
                            .with_message(format!(
                                "`{}` was referenced, but it is not defined.",
                                reg.inner.name
                            ))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish(),
            ),
        }
    }
}
