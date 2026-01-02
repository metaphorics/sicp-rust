// lang-rust.js - Rust language extension for Google Code Prettify
// For use with SICP Rust modernization project.
//
// (c) 2025, GNU GPL v.3.

/**
 * @fileoverview
 * Registers a language handler for Rust.
 *
 * Based on the Rust language specification.
 * https://doc.rust-lang.org/reference/
 */
PR['registerLangHandler'](
    PR['createSimpleLexer'](
        [
          // Whitespace
          [PR['PR_PLAIN'], /^[\t\n\r \xA0]+/, null, '\t\n\r \xA0'],
          // String literals (including raw strings)
          [PR['PR_STRING'], /^r#*"[\s\S]*?"#*/, null],
          [PR['PR_STRING'], /^"(?:[^"\\]|\\[\s\S])*"/, null, '"'],
          // Character literals
          [PR['PR_STRING'], /^'(?:[^'\\]|\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]{1,6}\}|.))'/, null, "'"],
          // Byte string and byte literals
          [PR['PR_STRING'], /^b"(?:[^"\\]|\\[\s\S])*"/, null],
          [PR['PR_STRING'], /^b'(?:[^'\\]|\\(?:x[0-9A-Fa-f]{2}|.))'/, null]
        ],
        [
          // Line comments
          [PR['PR_COMMENT'], /^\/\/[^\r\n]*/],
          // Block comments (non-nested for simplicity)
          [PR['PR_COMMENT'], /^\/\*[\s\S]*?\*\//],
          // Doc comments
          [PR['PR_COMMENT'], /^\/\/\/[^\r\n]*/],
          [PR['PR_COMMENT'], /^\/\/![^\r\n]*/],
          [PR['PR_COMMENT'], /^\/\*\*[\s\S]*?\*\//],
          [PR['PR_COMMENT'], /^\/\*![\s\S]*?\*\//],

          // Keywords
          [PR['PR_KEYWORD'], /^(?:as|async|await|break|const|continue|crate|dyn|else|enum|extern|false|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|true|type|unsafe|use|where|while)\b/],

          // Reserved keywords (future use)
          [PR['PR_KEYWORD'], /^(?:abstract|become|box|do|final|macro|override|priv|try|typeof|unsized|virtual|yield)\b/],

          // Primitive types
          [PR['PR_TYPE'], /^(?:bool|char|f32|f64|i8|i16|i32|i64|i128|isize|str|u8|u16|u32|u64|u128|usize)\b/],

          // Common standard library types
          [PR['PR_TYPE'], /^(?:String|Vec|Option|Result|Box|Rc|Arc|Cell|RefCell|Mutex|RwLock|HashMap|HashSet|BTreeMap|BTreeSet|VecDeque|LinkedList|BinaryHeap|Cow|Pin|PhantomData)\b/],

          // Lifetime annotations
          [PR['PR_ATTRIB_NAME'], /^'[a-z_][a-z0-9_]*/i],

          // Attributes
          [PR['PR_ATTRIB_NAME'], /^#!?\[[\s\S]*?\]/],

          // Macros (ending with !)
          [PR['PR_LITERAL'], /^[a-z_][a-z0-9_]*!/i],

          // Numeric literals
          // Hex
          [PR['PR_LITERAL'], /^0x[0-9a-f_]+(?:i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?/i],
          // Octal
          [PR['PR_LITERAL'], /^0o[0-7_]+(?:i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?/i],
          // Binary
          [PR['PR_LITERAL'], /^0b[01_]+(?:i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?/i],
          // Float
          [PR['PR_LITERAL'], /^[0-9][0-9_]*(?:\.[0-9][0-9_]*)?(?:[eE][+-]?[0-9_]+)?(?:f32|f64)?/],
          // Integer
          [PR['PR_LITERAL'], /^[0-9][0-9_]*(?:i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?/],

          // Identifiers (including type names starting with uppercase)
          [PR['PR_TYPE'], /^[A-Z][a-zA-Z0-9_]*/],
          [PR['PR_PLAIN'], /^[a-z_][a-zA-Z0-9_]*/i],

          // Operators and punctuation
          [PR['PR_PUNCTUATION'], /^(?:->|=>|::|\.\.\.?|\.\.=|\?\.|[+\-*\/%&|^!<>=]=?|<<|>>|&&|\|\|)/]
        ]),
    ['rs', 'rust']);
