# SICP Rust Modernization - Development Commands

set shell := ["bash", "-cu"]

# Default: show available commands
default:
    @just --list

# === Build Commands ===

# Full build: HTML5 + EPUB3
build:
    make

# Generate HTML5 only
html:
    make html

# Generate EPUB3 only
epub:
    make epub

# === Rust Examples ===

# Test all Rust examples
test:
    cd rust-examples && cargo test --workspace

# Test specific chapter (e.g., just test-chapter 1)
test-chapter chapter:
    cd rust-examples && cargo test -p chapter{{chapter}}

# === Formatting ===

# Format all Rust code
fmt:
    cd rust-examples && cargo fmt --all

# Check Rust formatting without changes
fmt-check:
    cd rust-examples && cargo fmt --all -- --check

# === Linting ===

# Run clippy on all Rust examples
lint:
    cd rust-examples && cargo clippy --workspace -- -D warnings

# Run clippy with fixes
lint-fix:
    cd rust-examples && cargo clippy --workspace --fix --allow-dirty

# === Quality ===

# Run all checks (fmt, lint, test)
check: fmt-check lint test

# Format and run all checks
ci: fmt lint test

# === Documentation ===

# Generate Rust documentation
docs:
    cd rust-examples && cargo doc --workspace --no-deps --open

# === Utilities ===

# Clean build artifacts
clean:
    cd rust-examples && cargo clean
    rm -f html/*.bak

# Watch for changes and rebuild (requires cargo-watch)
watch:
    cd rust-examples && cargo watch -x test

# Count lines in Texinfo source
loc:
    @wc -l sicp-pocket.texi

# Show chapter statistics
stats:
    @echo "Texinfo 소스:"
    @wc -l sicp-pocket.texi
    @echo "\nHTML 출력 파일:"
    @ls -1 html/*.xhtml | wc -l
    @echo "\nSVG 다이어그램:"
    @fd -e svg . html/fig/ | wc -l
    @echo "\nRust 예제:"
    @tokei rust-examples/ 2>/dev/null || echo "(상세 통계를 위해 tokei를 설치하세요)"
