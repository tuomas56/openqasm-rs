use crate::ast::{Decl, Expr, Program, Reg, Span, Stmt, Symbol};
use pretty::{DocAllocator, DocBuilder, Pretty};

impl<'a, D: DocAllocator<'a>, T: Pretty<'a, D>> Pretty<'a, D> for Span<T> {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        self.inner.pretty(alloc)
    }
}

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Symbol {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        alloc.text(self.to_string())
    }
}

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Reg {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        let doc = self.name.pretty(alloc);
        if let Some(index) = self.index {
            doc.append(alloc.text(index.to_string()).brackets())
        } else {
            doc
        }
    }
}

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Expr {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        fn to_pretty<'a, D: DocAllocator<'a>>(
            expr: Expr,
            prec: usize,
            left: bool,
            top: bool,
            alloc: &'a D,
        ) -> DocBuilder<'a, D> {
            match expr {
                Expr::Pi => alloc.text("pi"),
                Expr::Real(f) => alloc.text(f.to_string()),
                Expr::Int(i) => alloc.text(i.to_string()),
                Expr::Var(s) => s.pretty(alloc),

                Expr::Add(_, _)
                | Expr::Sub(_, _)
                | Expr::Mul(_, _)
                | Expr::Div(_, _)
                | Expr::Pow(_, _) => {
                    let (a, b, nprec, opname, assoc, comm) = match expr {
                        Expr::Add(a, b) => (a, b, 0, "+", true, true),
                        Expr::Sub(a, b) => (a, b, 0, "-", true, false),
                        Expr::Mul(a, b) => (a, b, 1, "*", true, true),
                        Expr::Div(a, b) => (a, b, 1, "/", true, false),
                        Expr::Pow(a, b) => (a, b, 3, "^", false, false),
                        _ => unreachable!(),
                    };

                    let a = to_pretty(*a.inner, nprec, true, false, alloc);
                    let b = to_pretty(*b.inner, nprec, false, false, alloc);
                    let out = a
                        .append(alloc.line())
                        .append(alloc.text(opname))
                        .append(alloc.space())
                        .append(b);

                    if nprec < prec || (nprec == prec && left != assoc && !comm && !top) {
                        out.parens().group()
                    } else {
                        out.group()
                    }
                }

                Expr::Neg(a) => {
                    let a = to_pretty(*a.inner, 2, false, false, alloc);
                    alloc.text("-").append(a)
                }

                Expr::Sin(_)
                | Expr::Cos(_)
                | Expr::Tan(_)
                | Expr::Ln(_)
                | Expr::Exp(_)
                | Expr::Sqrt(_) => {
                    let (a, opname) = match expr {
                        Expr::Sin(a) => (a, "sin"),
                        Expr::Cos(a) => (a, "cos"),
                        Expr::Tan(a) => (a, "tan"),
                        Expr::Ln(a) => (a, "ln"),
                        Expr::Exp(a) => (a, "exp"),
                        Expr::Sqrt(a) => (a, "sqrt"),
                        _ => unreachable!(),
                    };

                    let a = to_pretty(*a.inner, 0, true, true, alloc);
                    alloc.text(opname).append(a.parens())
                }
            }
        }

        to_pretty(self, 0, true, true, alloc)
    }
}

#[derive(Clone)]
struct Marker(&'static str, bool);

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Marker {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        alloc.text(self.0).append(if self.1 {
            alloc.hardline()
        } else {
            alloc.line()
        })
    }
}

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Stmt {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        match self {
            Stmt::U {
                theta,
                phi,
                lambda,
                reg,
            } => alloc
                .text("U")
                .append(
                    alloc
                        .line_()
                        .append(alloc.intersperse(
                            [theta.pretty(alloc), phi.pretty(alloc), lambda.pretty(alloc)],
                            Marker(",", false),
                        ))
                        .nest(2)
                        .append(alloc.line_())
                        .parens()
                        .group(),
                )
                .append(alloc.space())
                .append(reg.pretty(alloc))
                .append(alloc.text(";")),
            Stmt::CX { copy, xor } => alloc
                .text("CX")
                .append(alloc.space())
                .append(copy.pretty(alloc))
                .append(alloc.text(", "))
                .append(xor.pretty(alloc))
                .append(alloc.text(";")),
            Stmt::Reset { reg } => alloc
                .text("reset")
                .append(alloc.space())
                .append(reg.pretty(alloc))
                .append(alloc.text(";")),
            Stmt::Measure { from, to } => alloc
                .text("measure")
                .append(alloc.space())
                .append(from.pretty(alloc))
                .append(alloc.text(" -> "))
                .append(to.pretty(alloc))
                .append(alloc.text(";")),
            Stmt::Conditional { reg, val, then } => alloc
                .text("if")
                .append(
                    alloc
                        .line_()
                        .append(
                            reg.pretty(alloc)
                                .append(alloc.text(" == "))
                                .append(alloc.text(val.inner.to_string())),
                        )
                        .nest(2)
                        .append(alloc.line_())
                        .parens()
                        .group(),
                )
                .append(then.pretty(alloc)),
            Stmt::Barrier { regs } => alloc
                .text("barrier")
                .append(if regs.len() > 3 {
                    alloc.hardline()
                } else {
                    alloc.space()
                })
                .append(
                    alloc
                        .intersperse(
                            regs.chunks(3).map(|regs| {
                                alloc
                                    .intersperse(regs.iter().cloned(), Marker(",", false))
                                    .group()
                            }),
                            Marker(",", true),
                        )
                        .nest(2),
                )
                .append(alloc.text(";")),
            Stmt::Gate { name, params, args } => {
                let mut out = name.pretty(alloc);
                if !params.is_empty() {
                    let start = || {
                        if params.len() > 3 {
                            alloc.hardline()
                        } else {
                            alloc.line_()
                        }
                    };
                    out = out.append(
                        start()
                            .append(alloc.intersperse(
                                params.chunks(3).map(|regs| {
                                    alloc
                                        .intersperse(regs.iter().cloned(), Marker(",", false))
                                        .group()
                                }),
                                Marker(",", true),
                            ))
                            .nest(2)
                            .append(start())
                            .parens()
                            .group(),
                    );
                }

                if !args.is_empty() {
                    let start = if args.len() > 3 {
                        alloc.hardline()
                    } else {
                        alloc.line_()
                    };
                    out = out.append(alloc.space()).append(
                        start
                            .append(alloc.intersperse(
                                args.chunks(3).map(|regs| {
                                    alloc
                                        .intersperse(regs.iter().cloned(), Marker(",", false))
                                        .group()
                                }),
                                Marker(",", true),
                            ))
                            .nest(2)
                            .group(),
                    );
                }

                out.append(alloc.text(";"))
            }
        }
    }
}

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Decl {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        match self {
            Decl::CReg { reg } => reg.pretty(alloc).enclose("creg ", ";"),
            Decl::QReg { reg } => reg.pretty(alloc).enclose("qreg ", ";"),
            Decl::Include { file } => file.pretty(alloc).enclose("include \"", "\";"),
            Decl::Stmt(stmt) => stmt.pretty(alloc),
            Decl::Def {
                name,
                params,
                args,
                body,
            } => {
                let mut out = alloc
                    .text(if body.is_some() { "gate " } else { "opaque " })
                    .append(name);

                if !params.is_empty() {
                    out = out.append(
                        alloc
                            .line_()
                            .append(alloc.intersperse(params.into_iter(), Marker(",", false)))
                            .append(alloc.line_())
                            .parens()
                            .group(),
                    );
                }

                if !args.is_empty() {
                    out = out.append(alloc.space()).append(
                        alloc
                            .intersperse(args.into_iter(), Marker(",", false))
                            .group(),
                    );
                }

                if let Some(body) = body {
                    out.append(alloc.space()).append(
                        alloc
                            .hardline()
                            .append(alloc.intersperse(body, Marker("", true)))
                            .nest(2)
                            .append(alloc.hardline())
                            .braces(),
                    )
                } else {
                    out.append(";")
                }
            }
        }
    }
}

impl<'a, D: DocAllocator<'a>> Pretty<'a, D> for Program {
    fn pretty(self, alloc: &'a D) -> DocBuilder<'a, D> {
        let mut prev = true;
        let mut out = alloc
            .text("OPENQASM 2.0;")
            .append(alloc.hardline())
            .append(alloc.hardline());
        for decl in self.decls {
            let multi = matches!(&*decl.inner, Decl::Def { .. });

            if !prev && multi {
                out = out.append(alloc.hardline());
            }

            out = out.append(decl).append(alloc.hardline());

            if multi {
                out = out.append(alloc.hardline());
            }

            prev = multi;
        }
        out
    }
}

impl Reg {
    /// Pretty-print this object to a string.
    /// For more fine-grained control, use the `pretty::Pretty` trait from
    /// the `pretty` crate directly.
    pub fn to_pretty(&self, width: usize) -> String {
        let mut arena = pretty::Arena::new();
        let doc = self.clone().pretty(&mut arena).into_doc();
        let mut out = String::new();
        doc.render_fmt(width, &mut out).unwrap();
        out
    }
}

impl Expr {
    /// Pretty-print this object to a string.
    /// For more fine-grained control, use the `pretty::Pretty` trait from
    /// the `pretty` crate directly.
    pub fn to_pretty(&self, width: usize) -> String {
        let mut arena = pretty::Arena::new();
        let doc = self.clone().pretty(&mut arena).into_doc();
        let mut out = String::new();
        doc.render_fmt(width, &mut out).unwrap();
        out
    }
}

impl Stmt {
    /// Pretty-print this object to a string.
    /// For more fine-grained control, use the `pretty::Pretty` trait from
    /// the `pretty` crate directly.
    pub fn to_pretty(&self, width: usize) -> String {
        let mut arena = pretty::Arena::new();
        let doc = self.clone().pretty(&mut arena).into_doc();
        let mut out = String::new();
        doc.render_fmt(width, &mut out).unwrap();
        out
    }
}

impl Decl {
    /// Pretty-print this object to a string.
    /// For more fine-grained control, use the `pretty::Pretty` trait from
    /// the `pretty` crate directly.
    pub fn to_pretty(&self, width: usize) -> String {
        let mut arena = pretty::Arena::new();
        let doc = self.clone().pretty(&mut arena).into_doc();
        let mut out = String::new();
        doc.render_fmt(width, &mut out).unwrap();
        out
    }
}

impl Program {
    /// Pretty-print this object to a string.
    /// For more fine-grained control, use the `pretty::Pretty` trait from
    /// the `pretty` crate directly.
    pub fn to_pretty(&self, width: usize) -> String {
        let mut arena = pretty::Arena::new();
        let doc = self.clone().pretty(&mut arena).into_doc();
        let mut out = String::new();
        doc.render_fmt(width, &mut out).unwrap();
        out
    }
}
