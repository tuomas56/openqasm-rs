use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

/// An interned string constant.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Symbol(Rc<str>);

impl Symbol {
    /// Clone the symbol to make a new symbol.
    /// This is relatively cheap, just a refcount increase.
    pub fn to_symbol(&self) -> Symbol {
        self.clone()
    }

    /// Return a reference to a symbol.
    /// Useful when you have Box<Symbol> to work through the Deref impl.
    pub fn as_symbol(&self) -> &Symbol {
        self
    }

    /// Get a string representation of this symbol.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

thread_local! {
    static INTERNER: Interner = Interner { inner: RefCell::new(HashSet::new()) };
}

struct Interner {
    inner: RefCell<HashSet<Rc<str>>>,
}

impl Interner {
    fn insert(&self, val: &str) -> Rc<str> {
        let mut inner = self.inner.borrow_mut();
        match inner.get(val) {
            Some(val) => val.clone(),
            None => {
                let val: Rc<str> = Rc::from(val);
                inner.insert(val.clone());
                val
            }
        }
    }
}

impl Symbol {
    /// Create a new symbol from string data by interning it.
    pub fn new<S: AsRef<str>>(val: S) -> Symbol {
        Symbol(INTERNER.with(|interner| interner.insert(val.as_ref())))
    }
}

/// Represents a span of code in a file.
#[derive(Debug, Copy, Clone)]
pub struct FileSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) file: usize,
}

impl FileSpan {
    /// Create an empty span with no content.
    pub(crate) fn empty() -> FileSpan {
        FileSpan {
            start: 0,
            end: 0,
            file: 0,
        }
    }
}

/// An error report.
///
/// This contains a message and various labels all defined
/// in terms of FileSpans. In order to print this, you need
/// a mutable reference to the SourceCache from which these
/// spans were created. If you provide the wrong one,
/// you will get panics or garbage output.
///
/// Example Usage:
/// ```rust
/// let mut cache = SourceCache::new();
/// // generate an error, e.g from parsing, making sure to
/// // use `cache` as the source cache.
/// let err: Report = ...;
/// // print the error to stderr:
/// err.eprint(&mut cache)
/// ```
#[cfg(feature = "ariadne")]
pub type Report = ariadne::Report<FileSpan>;

#[cfg(feature = "ariadne")]
impl ariadne::Span for FileSpan {
    type SourceId = usize;

    fn source(&self) -> &Self::SourceId {
        &self.file
    }

    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }
}
