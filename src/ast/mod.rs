mod utils;
pub use self::utils::*;

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

/// Represents a whole program with defintions and statements.
/// The definitions and declarations may cover more than one file.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Program {
    /// The declarations in this program.
    pub decls: Vec<Span<Decl>>,
}

/// A declaration of some kind.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Decl {
    /// An `include` statement. These are usually resolved by the parser.
    Include {
        /// The file to include.
        file: Span<Symbol>,
    },
    /// A quantum register declaration.
    QReg {
        /// The register name and size.
        reg: Span<Reg>,
    },
    /// A classical register declaration.
    CReg {
        /// The register name and size.
        reg: Span<Reg>,
    },
    /// A gate definition.
    Def {
        /// The gate name.
        name: Span<Symbol>,
        /// The names of parameters to take.
        params: Vec<Span<Symbol>>,
        /// The names of the arguments to take.
        args: Vec<Span<Symbol>>,
        /// The content of the definition.
        /// A value of `None` represents an opaque gate definition.
        body: Option<Vec<Span<Stmt>>>,
    },
    /// A top-level statement.
    Stmt(Span<Stmt>),
}

/// A statement that represents an action.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Stmt {
    /// Apply a universal single-qubit unitary to a register.
    U {
        theta: Span<Expr>,
        phi: Span<Expr>,
        lambda: Span<Expr>,
        reg: Span<Reg>,
    },
    /// Apply a CNOT gate between two registers.
    CX { copy: Span<Reg>, xor: Span<Reg> },
    /// Measure a quantum register and store the result in a classical one.
    Measure { from: Span<Reg>, to: Span<Reg> },
    /// Reset a quantum register to the |0> state.
    Reset { reg: Span<Reg> },
    /// Prohibit optimizations crossing this point.
    Barrier { regs: Vec<Span<Reg>> },
    /// Apply a defined gate to some qubits.
    Gate {
        name: Span<Symbol>,
        params: Vec<Span<Expr>>,
        args: Vec<Span<Reg>>,
    },
    /// Perform an action conditional on a classical register value.
    Conditional {
        reg: Span<Reg>,
        val: Span<usize>,
        then: Span<Stmt>,
    },
}

/// A parameter expression.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Expr {
    /// The circle constant Pi.
    Pi,
    /// An arbitrary real number.
    Real(f32),
    /// An integer.
    Int(usize),
    /// A defined parameter.
    Var(Symbol),
    /// The addition of two expressions.
    Add(Span<Expr>, Span<Expr>),
    /// The subtraction of two expressions.
    Sub(Span<Expr>, Span<Expr>),
    /// The multiplication of two expressions.
    Mul(Span<Expr>, Span<Expr>),
    /// The division of two expressions.
    Div(Span<Expr>, Span<Expr>),
    /// The exponentiation of two expressions.
    Pow(Span<Expr>, Span<Expr>),
    /// The negation of an expression.
    Neg(Span<Expr>),
    /// The sine of an expression.
    Sin(Span<Expr>),
    /// The cosine of an expression.
    Cos(Span<Expr>),
    /// The tangent of an expression.
    Tan(Span<Expr>),
    /// The exponential of an expression.
    Exp(Span<Expr>),
    /// The natural logarithm of an expression.
    Ln(Span<Expr>),
    /// The square root of an expression.
    Sqrt(Span<Expr>),
}

/// A reference to (or definition of) a register or qubit.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Reg {
    /// The name of the register.
    pub name: Symbol,
    /// The index to select if `Some` variant given,
    /// `None` represents the whole register.
    /// In definitions, this represents the size
    /// of the register, and `None` means size one.
    pub index: Option<usize>,
}

/// An object with an attached span.
/// The span references where in the source code
/// this object was derived from.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Span<T> {
    /// The span corresponding to this object.
    pub span: FileSpan,
    /// The actual object itself.
    pub inner: Box<T>,
}
