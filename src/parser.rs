use crate::ast::{Decl, FileSpan, Program, Report};
use ariadne::{Label, ReportKind, Source};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

lalrpop_mod!(
    #[allow(clippy::all)]
    parser_impl,
    "/grammar.rs"
);

/// A cache for source strings and files.
///
/// This is used by `Parser` to store source code, as well as by `Report` when printing.
/// Printing a report with a different cache than was used to create it may
/// result in panics and/or garbage output.
///
/// Example Usage:
/// ```rust
/// let mut cache = SourceCache::new();
/// // Do something that creates an error
/// let err: Report = ...
/// // Print it to stderr
/// err.eprint(&mut cache).unwrap();
/// ```
pub struct SourceCache {
    files: HashMap<PathBuf, usize>,
    paths: HashMap<usize, PathBuf>,
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
        let mut sources = HashMap::new();
        // We include the empty source in ID zero for errors which have no definite location.
        sources.insert(0, Source::from(" "));
        SourceCache {
            files: HashMap::new(),
            paths: HashMap::new(),
            sources,
            strings: HashMap::new(),
            next_id: 1,
        }
    }

    fn add_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<usize> {
        if let Some(id) = self.files.get(path.as_ref()) {
            Ok(*id)
        } else {
            use std::io::Read;

            // The file is not cached, read it from the filesystem.
            let mut file = std::fs::File::open(path.as_ref())?;

            let mut contents = String::new();
            file.read_to_string(&mut contents)?;

            // CRLF line endings break ariadne's formatting.
            contents = contents.replace("\r\n", "\n");
            if contents.is_empty() {
                contents.push(' ');
            }

            self.files.insert(path.as_ref().to_path_buf(), self.next_id);
            self.paths.insert(self.next_id, path.as_ref().to_path_buf());
            self.sources
                .insert(self.next_id, Source::from(contents.clone()));
            self.strings.insert(self.next_id, contents);

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

        self.sources
            .insert(self.next_id, Source::from(source.clone()));
        self.strings.insert(self.next_id, source);

        self.next_id += 1;
        self.next_id - 1
    }

    fn is_cached<P: AsRef<Path>>(&self, path: P) -> bool {
        self.files.contains_key(path.as_ref())
    }
}

impl ariadne::Cache<usize> for SourceCache {
    fn fetch(&mut self, id: &usize) -> Result<&Source, Box<dyn std::fmt::Debug + '_>> {
        Ok(&self.sources[id])
    }

    fn display<'a>(&self, id: &'a usize) -> Option<Box<dyn std::fmt::Display + 'a>> {
        if *id == 0 {
            None
        } else {
            Some(Box::new(self.paths.get(id)?.display().to_string()))
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
/// file relative to the current working directory.
pub enum FilePolicy<'a> {
    /// All requests result in an error.
    Deny,
    /// All requests are silently ignored.
    Ignore,
    /// Requests are made to the filesystem relative to the current directory.
    FileSystem,
    /// Handle the requests with a custom function.
    Custom(&'a mut dyn FnMut(&Path) -> FileResult),
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
/// ```rust
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
/// match parser.done() {
///     Ok(program) => ..., // do something with this
///     Err(errors) => for error in errors {
///         // print the error to stderr
///         error.eprint(&mut cache).unwrap();
///     }
/// }
/// ```
pub struct Parser<'a> {
    cache: &'a mut SourceCache,
    programs: HashMap<usize, Program>,
    errors: Vec<Report>,
    policy: FilePolicy<'a>,
}

impl<'a> Parser<'a> {
    /// Construct a new parser that will add source to the given cache.
    pub fn new(cache: &'a mut SourceCache) -> Parser<'a> {
        Parser {
            cache,
            programs: HashMap::new(),
            errors: Vec::new(),
            policy: FilePolicy::FileSystem,
        }
    }

    /// Set the file policy for this parser. The default is `FilePolicy::FileSystem`.
    pub fn with_file_policy(mut self, policy: FilePolicy<'a>) -> Self {
        self.policy = policy;
        self
    }

    /// Attempt to parse the file at the given path.
    pub fn parse_file<P: AsRef<Path>>(&mut self, path: P) {
        self.process_file(path, None);
    }

    /// Attempt to parse the given source code.
    /// If a path is given, this source will be associated with
    /// the given path in the cache, and any errors will show
    /// this as the file name.
    pub fn parse_source<P: AsRef<Path>>(&mut self, source: String, path: Option<P>) {
        let id = self.cache.add_source(source, path);
        self.parse_prog(id, true);
    }

    /// Stop the parser and return any errors encountered or the parsed AST.
    pub fn done(self) -> Result<Program, Vec<Report>> {
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

    fn process_file<P: AsRef<Path>>(&mut self, path: P, from: Option<FileSpan>) {
        match self.policy {
            FilePolicy::Deny => {
                let err = Report::build(ReportKind::Error, 0usize, 0).with_message(format!(
                    "File `{}` could not be read - sandboxing is enabled",
                    path.as_ref().display()
                ));

                self.errors.push(match from {
                    None => err.finish(),
                    Some(span) => err
                        .with_label(
                            Label::new(span)
                                .with_message("This file is included here")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish(),
                });
            }
            FilePolicy::Ignore => (),
            FilePolicy::FileSystem => {
                let res = self.cache.add_file(path.as_ref()).map_err(|e| match from {
                    None => Report::build(ReportKind::Error, 0usize, 0)
                        .with_message(format!(
                            "File `{}` could not be read - {}",
                            path.as_ref().display(),
                            e
                        ))
                        .finish(),
                    Some(span) => Report::build(ReportKind::Error, span.file, span.start)
                        .with_message(format!(
                            "File `{}` could not be read - {}",
                            path.as_ref().display(),
                            e
                        ))
                        .with_label(
                            Label::new(span)
                                .with_message("This file is included here.")
                                .with_color(ariadne::Color::Cyan),
                        )
                        .finish(),
                });

                match res {
                    Ok(id) => self.parse_prog(id, from.is_none()),
                    Err(e) => self.errors.push(e),
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
                    FileResult::Error(e) => {
                        let err =
                            Report::build(ReportKind::Error, 0usize, 0).with_message(format!(
                                "File `{}` could not be read - {}",
                                path.as_ref().display(),
                                e
                            ));

                        self.errors.push(match from {
                            None => err.finish(),
                            Some(span) => err
                                .with_label(
                                    Label::new(span)
                                        .with_message("This file is included here.")
                                        .with_color(ariadne::Color::Cyan),
                                )
                                .finish(),
                        });
                    }
                }
            }
        }
    }

    fn parse_prog(&mut self, id: usize, toplevel: bool) {
        if self.programs.contains_key(&id) {
            return;
        }

        let parse = if toplevel {
            // If we are at the top level file, then we would expect to have
            // a "OPENQASM 2.0" signature at the top,
            parser_impl::TopLevelParser::new().parse(id, &self.cache.strings[&id])
        } else {
            // but if this is an included file it may not be there,
            // so use a different parser.
            parser_impl::IncludedParser::new().parse(id, &self.cache.strings[&id])
        };

        let res = parse.map_err(|e| {
            use lalrpop_util::ParseError;
            match e {
                // An error occurred while parsing an integer or real.
                ParseError::User {
                    error: (reason, label, span),
                } => Report::build(ReportKind::Error, id, 0)
                    .with_message(reason)
                    .with_label(
                        Label::new(span)
                            .with_message(label)
                            .with_color(ariadne::Color::Cyan),
                    )
                    .finish(),
                // We got extra tokens after what was supposed to be EOF.
                // I don't think this ever occurs.
                ParseError::ExtraToken {
                    token: (start, tok, end),
                } => Report::build(ReportKind::Error, id, start)
                    .with_message(format!("Unexpected token `{}`", tok))
                    .with_label(
                        Label::new(FileSpan {
                            start,
                            end,
                            file: id,
                        })
                        .with_message("This token was unexpected")
                        .with_color(ariadne::Color::Cyan),
                    )
                    .finish(),
                // There is an invalid character sequence that does not match any token.
                ParseError::InvalidToken { location } => {
                    Report::build(ReportKind::Error, id, location)
                        .with_message("Invalid token")
                        .with_label(
                            Label::new(FileSpan {
                                start: location,
                                end: location,
                                file: id,
                            })
                            .with_message("This token is invalid")
                            .with_color(ariadne::Color::Cyan),
                        )
                        .finish()
                }
                // We had an EOF but we weren't done parsing.
                ParseError::UnrecognizedEOF { location, expected } => {
                    Report::build(ReportKind::Error, id, location)
                        .with_message("Unexpected end of file")
                        .with_label(
                            Label::new(FileSpan {
                                start: location,
                                end: location,
                                file: id,
                            })
                            .with_message(format!("Expected {} here", expected.join(" or ")))
                            .with_color(ariadne::Color::Cyan),
                        )
                        .finish()
                }
                // We expected something else here.
                ParseError::UnrecognizedToken {
                    token: (start, tok, end),
                    expected,
                } => Report::build(ReportKind::Error, id, start)
                    .with_message(format!("Unexpected token `{}`", tok))
                    .with_label(
                        Label::new(FileSpan {
                            start,
                            end,
                            file: id,
                        })
                        .with_message(format!("Expected {} here", expected.join(" or ")))
                        .with_color(ariadne::Color::Cyan),
                    )
                    .finish(),
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
                    self.process_file(parent.join(path), Some(span));
                }
            }
            // If no, then we assume this is the top-level path:
            None => {
                for (path, span) in includes {
                    self.process_file(path, Some(span));
                }
            }
        }
    }
}
