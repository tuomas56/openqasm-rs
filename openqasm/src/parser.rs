use crate::ast::{Decl, FileSpan, Program};
use logos::Logos;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[cfg(feature = "ariadne")]
use {
    crate::ast::Report,
    ariadne::{Label, ReportKind, Source},
};

mod generated {
    use crate::ast::Symbol;
    use logos::Logos;

    lalrpop_mod!(
        #[allow(clippy::all)]
        pub parser_impl,
        "/grammar.rs"
    );

    #[derive(Logos, Debug, PartialEq, Clone)]
    pub enum Token {
        #[token("OPENQASM 2.0")]
        OPENQASM,

        #[token("include")]
        Include,

        #[token("qreg")]
        QReg,

        #[token("creg")]
        CReg,

        #[token("gate")]
        Gate,

        #[token("opaque")]
        Opaque,

        #[token("measure")]
        Measure,

        #[token("reset")]
        Reset,

        #[token("barrier")]
        Barrier,

        #[token("if")]
        If,

        #[token("(")]
        LParen,

        #[token(")")]
        RParen,

        #[token("{")]
        LBrace,

        #[token("}")]
        RBrace,

        #[token("==")]
        Equals,

        #[token(";")]
        Semicolon,

        #[token(",")]
        Comma,

        #[token("U")]
        U,

        #[token("CX")]
        CX,

        #[token("pi")]
        Pi,

        #[token("sin")]
        Sin,

        #[token("cos")]
        Cos,

        #[token("tan")]
        Tan,

        #[token("exp")]
        Exp,

        #[token("ln")]
        Ln,

        #[token("sqrt")]
        Sqrt,

        #[token("^")]
        Pow,

        #[token("*")]
        Mul,

        #[token("/")]
        Div,

        #[token("+")]
        Add,

        #[token("-")]
        Sub,

        #[token("[")]
        LBracket,

        #[token("]")]
        RBracket,

        #[token("->")]
        Arrow,

        #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |tok| {
            Symbol::new(tok.slice())
        })]
        Identifier(Symbol),

        #[regex(r"[0-9]+", |tok| tok.slice().parse())]
        Integer(u64),

        #[regex(r"[0-9]+\.[0-9]+", |tok| tok.slice().parse())]
        Real(f32),

        #[regex(r#""[^"]*""#, |tok| {
            let slice = tok.slice();
            Symbol::new(&slice[1..slice.len()-1])
        })]
        String(Symbol),

        #[error]
        #[regex(r"[ \t\n\f]+", logos::skip)]
        #[regex(r"//[^\n\r]*[\n\r]*", logos::skip)]
        Error,
    }
}

/// A cache for source strings and files.
///
/// This is used by `Parser` to store source code, as well as by `Report` when printing.
/// Printing a report with a different cache than was used to create it may
/// result in panics and/or garbage output.
///
/// Example Usage:
/// ```ignore
/// let mut cache = SourceCache::new();
/// // Do something that creates an error
/// let err: Report = ...
/// // Print it to stderr
/// err.eprint(&mut cache).unwrap();
/// ```
pub struct SourceCache {
    files: HashMap<PathBuf, usize>,
    paths: HashMap<usize, PathBuf>,
    #[cfg(feature = "ariadne")]
    sources: HashMap<usize, Source>,
    strings: HashMap<usize, String>,
    next_id: usize,
}

impl Default for SourceCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceCache {
    /// Create an empty cache.
    pub fn new() -> SourceCache {
        #[cfg(feature = "ariadne")]
        let sources = {
            let mut sources = HashMap::new();
            // We include the empty source in ID zero for errors which have no definite location.
            sources.insert(0, Source::from(" "));
            sources
        };

        SourceCache {
            files: HashMap::new(),
            paths: HashMap::new(),
            #[cfg(feature = "ariadne")]
            sources,
            strings: HashMap::new(),
            next_id: 1,
        }
    }

    /// Get the source code that is referred to by a `FileSpan`
    pub fn get_source(&self, span: FileSpan) -> &str {
        &self.strings[&span.file][span.start..span.end]
    }

    /// Get the path of the file that this `FileSpan` belongs to.
    pub fn get_path(&self, span: FileSpan) -> Option<&Path> {
        self.paths.get(&span.file).map(PathBuf::as_path)
    }

    fn add_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<usize> {
        let path = path.as_ref().to_path_buf();
        if let Some(id) = self.files.get(&path) {
            Ok(*id)
        } else {
            use std::io::Read;

            // The file is not cached, read it from the filesystem.
            let mut file = std::fs::File::open(&path)?;

            let mut contents = String::new();
            file.read_to_string(&mut contents)?;

            // CRLF line endings break ariadne's formatting.
            contents = contents.replace("\r\n", "\n");
            if contents.is_empty() {
                contents.push(' ');
            }

            self.files.insert(path.clone(), self.next_id);
            self.paths.insert(self.next_id, path);
            self.strings.insert(self.next_id, contents.clone());

            #[cfg(feature = "ariadne")]
            self.sources.insert(self.next_id, Source::from(contents));

            self.next_id += 1;
            Ok(self.next_id - 1)
        }
    }

    fn add_source<P: AsRef<Path>>(&mut self, source: String, path: Option<P>) -> usize {
        if let Some(path) = path {
            // If a path is defined, we might have this cached.
            if let Some(id) = self.files.get(path.as_ref()) {
                return *id;
            }

            self.files.insert(path.as_ref().to_path_buf(), self.next_id);
            self.paths.insert(self.next_id, path.as_ref().to_path_buf());
        }

        self.strings.insert(self.next_id, source.clone());

        #[cfg(feature = "ariadne")]
        self.sources.insert(self.next_id, Source::from(source));

        self.next_id += 1;
        self.next_id - 1
    }

    fn is_cached<P: AsRef<Path>>(&self, path: P) -> bool {
        self.files.contains_key(path.as_ref())
    }
}

#[cfg(feature = "ariadne")]
impl ariadne::Cache<usize> for SourceCache {
    fn fetch(&mut self, id: &usize) -> Result<&Source, Box<dyn std::fmt::Debug + '_>> {
        Ok(&self.sources[id])
    }

    fn display<'a>(&self, id: &'a usize) -> Option<Box<dyn std::fmt::Display + 'a>> {
        if *id == 0 {
            None
        } else {
            let path = self.paths.get(id)?;
            Some(Box::new(
                // If the path is absolute, attempt to display it relative
                // to the working directory.
                path.is_absolute()
                    .then(|| ())
                    .and_then(|()| {
                        std::env::current_dir()
                            .and_then(|dir| dir.canonicalize())
                            .ok()
                            .and_then(|base| path.strip_prefix(base).ok())
                    })
                    // If there is an error, just display the whole path.
                    .unwrap_or(path)
                    .display()
                    .to_string(),
            ))
        }
    }
}

/// The result of a custom file query.
#[derive(Debug)]
pub enum FileResult {
    /// A successful query, and the provided source should be parsed.
    Success(String),
    /// A successful query, but there is no source to parse.
    Ignore,
    /// The query was unsuccessful.
    Error(Box<dyn std::error::Error>),
}

/// The action to take when a file is requested by the parser.
///
/// The default choice is `FileSystem`, which loads the requested
/// file relative to the current working directory. It also includes
/// a list of files to hardcode. By default this is "qelib1.inc".
pub enum FilePolicy<'a> {
    /// All requests result in an error.
    Deny,
    /// All requests except top-level ones are silently ignored.
    Ignore,
    /// Requests are made to the filesystem relative to the current directory.
    FileSystem {
        /// The following files are hardcoded
        /// and should be used if the same path does not exist locally.
        /// By default, this is just "qelib1.inc". Please note that
        /// the paths must be matched exactly in include statements.
        hardcoded: HashMap<PathBuf, String>,
    },
    /// Handle the requests with a custom function.
    Custom(&'a mut dyn FnMut(&Path) -> FileResult),
}

impl<'a> FilePolicy<'a> {
    /// Create `FilePolicy::FileSystem` with "qelib1.inc" hardcoded.
    pub fn filesystem() -> FilePolicy<'a> {
        let mut hardcoded = HashMap::new();
        hardcoded.insert(
            PathBuf::from("qelib1.inc"),
            include_str!("../includes/qelib1.inc").to_string(),
        );
        FilePolicy::FileSystem { hardcoded }
    }

    /// If this is a `FileSystem` variant, add a file to the hardcoded list.
    /// If this is any other variant, panic.
    pub fn with_file<P: AsRef<Path>>(mut self, path: P, source: &str) -> Self {
        match self {
            FilePolicy::FileSystem { ref mut hardcoded } => {
                hardcoded.insert(path.as_ref().to_path_buf(), source.to_string());
                self
            }
            _ => panic!("Can't add a hardcoded file to a non-FileSystem FilePolicy."),
        }
    }
}

/// Parser for OpenQASM 2.0 programs.
///
/// `Parser` can process programs from both files and source strings,
/// and will attempt to resolve `include` statements to other files.
///
/// A file policy can be set which dictates how `Parser` reacts to
/// a request to parse a file. The default choice is `FilePolicy::FileSystem`
/// which attempts to read the file from disk, but this can be changed
/// using `Parser::with_file_policy`.
///
/// You can use `Parser::done` to stop the parser and return a list of
/// errors encountered while parsing. The parser does not attempt to
/// do error recovery, so there will be at most one parse error per file
/// or `include` statement.
///
/// Example Usage:
/// ```ignore
/// let mut cache = SourceCache::new();
/// let mut parser = Parser::new(&mut cache);
///
/// parser.parse_file("test.qasm");
/// parser.parse_source("
///     OPENQASM 2.0;
///     qreg a;
///     creg b;
///     cx a, b;
/// ");
///
/// match parser.done().to_errors() {
///     Ok(program) => ..., // do something with this
///     Err(errors) => errors.print(&mut cache).unwrap()
/// }
/// ```
pub struct Parser<'a> {
    cache: &'a mut SourceCache,
    programs: HashMap<usize, Program>,
    errors: Vec<ParseError>,
    policy: FilePolicy<'a>,
}

impl<'a> Parser<'a> {
    /// Construct a new parser that will add source to the given cache.
    pub fn new(cache: &'a mut SourceCache) -> Parser<'a> {
        Parser {
            cache,
            programs: HashMap::new(),
            errors: Vec::new(),
            policy: FilePolicy::filesystem(),
        }
    }

    /// Set the file policy for this parser. The default is `FilePolicy::FileSystem`.
    pub fn with_file_policy(mut self, policy: FilePolicy<'a>) -> Self {
        self.policy = policy;
        self
    }

    /// Attempt to parse the file at the given path.
    pub fn parse_file<P: AsRef<Path>>(&mut self, path: P) {
        self.process_file(path, Path::new("."), None);
    }

    /// Attempt to parse the given source code.
    /// If a path is given, this source will be associated with
    /// the given path in the cache, and any errors will show
    /// this as the file name.
    pub fn parse_source<P: AsRef<Path>>(&mut self, source: String, path: Option<P>) {
        let id = self.cache.add_source(source, path);
        self.parse_prog(id, false);
    }

    /// Stop the parser and return any errors encountered or the parsed AST.
    pub fn done(self) -> Result<Program, Vec<ParseError>> {
        if self.errors.is_empty() {
            Ok(Program {
                decls: self
                    .programs
                    .into_iter()
                    .flat_map(|(_, prog)| prog.decls)
                    .collect(),
            })
        } else {
            Err(self.errors)
        }
    }

    fn process_file<P: AsRef<Path>>(&mut self, path: P, parent: &Path, from: Option<FileSpan>) {
        match self.policy {
            FilePolicy::Deny => self.errors.push(ParseError::ReadUnableSandboxed {
                path: path.as_ref().to_path_buf(),
                from,
            }),
            FilePolicy::Ignore => {
                if from.is_none() {
                    match parent
                        .join(&path)
                        .canonicalize()
                        .and_then(|path| self.cache.add_file(path))
                        .map_err(|error| ParseError::ReadUnableFilesystem {
                            path: path.as_ref().to_path_buf(),
                            from,
                            error,
                        }) {
                        Ok(id) => self.parse_prog(id, from.is_none()),
                        Err(e) => self.errors.push(e),
                    }
                }
            }
            FilePolicy::FileSystem { ref mut hardcoded } => {
                if let Some(source) = hardcoded.get(path.as_ref()) {
                    match parent
                        .join(&path)
                        .canonicalize()
                        .and_then(|path| self.cache.add_file(path))
                    {
                        Ok(id) => self.parse_prog(id, from.is_none()),
                        Err(_) => {
                            let source = source.clone();
                            let id = self.cache.add_source(source, Some(path));
                            self.parse_prog(id, false);
                        }
                    }
                } else {
                    match parent
                        .join(&path)
                        .canonicalize()
                        .and_then(|path| self.cache.add_file(path))
                        .map_err(|error| ParseError::ReadUnableFilesystem {
                            path: path.as_ref().to_path_buf(),
                            from,
                            error,
                        }) {
                        Ok(id) => self.parse_prog(id, from.is_none()),
                        Err(e) => self.errors.push(e),
                    }
                }
            }
            FilePolicy::Custom(ref mut func) => {
                if self.cache.is_cached(path.as_ref()) {
                    return;
                }

                match func(path.as_ref()) {
                    FileResult::Ignore => (),
                    FileResult::Success(source) => {
                        let id = self.cache.add_source(source, Some(path));
                        self.parse_prog(id, from.is_none());
                    }
                    FileResult::Error(error) => self.errors.push(ParseError::ReadUnableCustom {
                        path: path.as_ref().to_path_buf(),
                        from,
                        error,
                    }),
                }
            }
        }
    }

    fn parse_prog(&mut self, id: usize, toplevel: bool) {
        if self.programs.contains_key(&id) {
            return;
        }

        let lexer = generated::Token::lexer(&self.cache.strings[&id])
            .spanned()
            .map(|(tok, span)| match tok {
                generated::Token::Error => Err(FileSpan {
                    start: span.start,
                    end: span.end,
                    file: id,
                }),
                _ => Ok((span.start, tok, span.end)),
            });

        let parse = if toplevel {
            // If we are at the top level file, then we would expect to have
            // a "OPENQASM 2.0" signature at the top,
            generated::parser_impl::TopLevelParser::new().parse(id, lexer)
        } else {
            // but if this is an included file it may not be there,
            // so use a different parser.
            generated::parser_impl::IncludedParser::new().parse(id, lexer)
        };

        let res = parse.map_err(|e| {
            use lalrpop_util::ParseError as PE;
            match e {
                PE::User { error: span } => ParseError::InvalidToken { span },
                // We got extra tokens after what was supposed to be EOF.
                // I don't think this ever occurs.
                PE::ExtraToken {
                    token: (start, tok, end),
                } => ParseError::UnexpectedToken {
                    token: format!("{tok:?}"),
                    expected: Vec::new(),
                    span: FileSpan {
                        start,
                        end,
                        file: id,
                    },
                },
                // There is an invalid character sequence that does not match any token.
                PE::InvalidToken { location } => ParseError::InvalidToken {
                    span: FileSpan {
                        start: location,
                        end: location,
                        file: id,
                    },
                },
                // We had an EOF but we weren't done parsing.
                PE::UnrecognizedEOF { location, expected } => ParseError::UnexpectedEOF {
                    expected,
                    span: FileSpan {
                        start: location,
                        end: location,
                        file: id,
                    },
                },
                // We expected something else here.
                PE::UnrecognizedToken {
                    token: (start, tok, end),
                    expected,
                } => ParseError::UnexpectedToken {
                    expected,
                    token: format!("{tok:?}"),
                    span: FileSpan {
                        start,
                        end,
                        file: id,
                    },
                },
            }
        });

        match res {
            Ok(program) => {
                self.programs.insert(id, program);
                self.find_includes(id);
            }
            Err(e) => self.errors.push(e),
        }
    }

    fn find_includes(&mut self, id: usize) {
        let prog = &self.programs[&id];
        // Gather all the include statements for this program:
        let mut includes = Vec::new();
        for decl in &prog.decls {
            if let Decl::Include { file } = &*decl.inner {
                includes.push((file.inner.to_string(), file.span));
            }
        }

        // Check if this program has a path defined for its source code:
        match self.cache.paths.get(&id) {
            // If yes, then we want to find a file relative to this path.
            Some(path) => {
                let parent = path.parent().unwrap().to_path_buf();
                for (path, span) in includes {
                    self.process_file(path, &parent, Some(span));
                }
            }
            // If no, then we assume this is the top-level path:
            None => {
                for (path, span) in includes {
                    self.process_file(path, Path::new("."), Some(span));
                }
            }
        }
    }
}

/// An error produced during parsing.
///
/// This includes two types of errors:
/// * `ReadUnable...` means a file couldn't be read
/// while trying to include or open it,
/// * Everything else is generated from the actual parsing itself.
///
#[derive(Debug, Error)]
pub enum ParseError {
    /// Can't read the file, the parser is sandboxed.
    #[error("can't read file - sandboxed")]
    ReadUnableSandboxed {
        path: PathBuf,
        from: Option<FileSpan>,
    },
    /// Can't read the file, the filesystem had an error.
    #[error("can't read file - io error")]
    ReadUnableFilesystem {
        path: PathBuf,
        from: Option<FileSpan>,
        #[source]
        error: std::io::Error,
    },
    /// Can't read the file, the custom handler had an error.
    #[error("can't read file - custom error")]
    ReadUnableCustom {
        path: PathBuf,
        from: Option<FileSpan>,
        #[source]
        error: Box<dyn std::error::Error>,
    },
    /// The token at this location is invalid.
    #[error("invalid token")]
    InvalidToken { span: FileSpan },
    /// The token at this location was unexpected.
    #[error("unexpected token")]
    UnexpectedToken {
        token: String,
        expected: Vec<String>,
        span: FileSpan,
    },
    /// A token was expected at this location.
    #[error("unexpected eof")]
    UnexpectedEOF {
        expected: Vec<String>,
        span: FileSpan,
    },
}

#[cfg(feature = "ariadne")]
impl ParseError {
    /// Convert this error into a `Report` for printing.
    pub fn to_report(&self) -> Report {
        match self {
            ParseError::ReadUnableSandboxed { path, from } => {
                let err = Report::build(ReportKind::Error, 0usize, 0).with_message(format!(
                    "File `{}` could not be read - sandboxing is enabled",
                    path.display()
                ));

                match from {
                    None => err.finish(),
                    Some(span) => err
                        .with_label(
                            Label::new(*span)
                                .with_message("This file is included here")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish(),
                }
            }
            ParseError::ReadUnableFilesystem { path, error, from } => {
                let err = Report::build(ReportKind::Error, 0usize, 0).with_message(format!(
                    "File `{}` could not be read - {}",
                    path.display(),
                    error
                ));

                match from {
                    None => err.finish(),
                    Some(span) => err
                        .with_label(
                            Label::new(*span)
                                .with_message("This file is included here.")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish(),
                }
            }
            ParseError::ReadUnableCustom { path, error, from } => {
                let err = Report::build(ReportKind::Error, 0usize, 0).with_message(format!(
                    "File `{}` could not be read - {}",
                    path.display(),
                    error
                ));

                match from {
                    None => err.finish(),
                    Some(span) => err
                        .with_label(
                            Label::new(*span)
                                .with_message("This file is included here.")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish(),
                }
            }
            ParseError::UnexpectedToken {
                token,
                expected,
                span,
            } => {
                if expected.is_empty() {
                    Report::build(ReportKind::Error, span.file, span.start)
                        .with_message(format!("Unexpected token `{}`", token))
                        .with_label(
                            Label::new(*span)
                                .with_message("This token was unexpected")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish()
                } else {
                    Report::build(ReportKind::Error, span.file, span.start)
                        .with_message(format!("Unexpected token `{}`", token))
                        .with_label(
                            Label::new(*span)
                                .with_message(format!("Expected {} here", expected.join(" or ")))
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish()
                }
            }
            ParseError::UnexpectedEOF { expected, span } => {
                Report::build(ReportKind::Error, span.file, span.start)
                    .with_message("Unexpected end of file")
                    .with_label(
                        Label::new(*span)
                            .with_message(format!("Expected {} here", expected.join(" or ")))
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish()
            }
            ParseError::InvalidToken { span } => {
                Report::build(ReportKind::Error, span.file, span.start)
                    .with_message("Invalid token")
                    .with_label(
                        Label::new(*span)
                            .with_message("This token is invalid")
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish()
            }
        }
    }
}
