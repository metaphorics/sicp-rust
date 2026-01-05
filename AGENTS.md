# AGENTS

## Requirements

- Translate `sicp-pocket.texi`, `exercises.texi`, `figures.texi`, and all content under `rust-examples/` fully to Korean with natural translation.
- For every occurrence of translated terms, include the English original in parentheses.
- Include comments and bibliography in the translation scope.
- Translate code and literals (do not keep them in English), while preserving Texinfo commands, `@ref` keys, and node identifiers.
- Apply to all chapters.

## Execution Notes

- Preserve Texinfo syntax and URLs.
- `exercises.texi` and `figures.texi` are generated; changes may need re-application if regenerated.

## Plan (Batching)

1. Update `glossaries.md` with global translation rules.
2. Translate front matter and history comments.
3. Translate chapters sequentially, including exercises and figures lists.
4. Translate all `rust-examples/` source and Markdown docs.
5. Translate References and remaining comments.
6. Verify with `make html` and Rust build/tests if requested.
