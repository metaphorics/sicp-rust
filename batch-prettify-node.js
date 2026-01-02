#!/usr/bin/env node

// Usage: ./batch-prettify-node.js file1 [file2 ...]
// Prettifies Scheme and Rust code in HTML files using jsdom.
// Replaces PhantomJS-based batch-prettify.js (PhantomJS deprecated 2018).

// (c) 2014 Andres Raba, GNU GPL v.3.
// (c) 2025 Updated for Node.js

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

// Get command-line arguments
const args = process.argv.slice(2);

if (args.length === 0) {
  console.log("Usage: ./batch-prettify-node.js file1 [file2 ...]");
  process.exit(0);
}

// Filter to existing files
const files = args.filter(file => {
  if (fs.existsSync(file)) {
    return true;
  } else {
    console.log('No such file: ' + file);
    return false;
  }
});

// Load prettify scripts
const highlightDir = path.join(__dirname, 'html', 'js', 'highlight');
const prettifyJs = fs.readFileSync(path.join(highlightDir, 'prettify.js'), 'utf8');
const langLispJs = fs.readFileSync(path.join(highlightDir, 'lang-lisp.js'), 'utf8');

// Load lang-rust.js if it exists
let langRustJs = '';
const langRustPath = path.join(highlightDir, 'lang-rust.js');
if (fs.existsSync(langRustPath)) {
  langRustJs = fs.readFileSync(langRustPath, 'utf8');
}

// Process a single file
async function processFile(filePath) {
  const html = fs.readFileSync(filePath, 'utf8');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    resources: 'usable',
    url: 'file://' + path.resolve(filePath)
  });

  const { window } = dom;
  const { document } = window;

  // Inject prettify scripts into the document
  const scriptContent = `
    ${prettifyJs}
    ${langLispJs}
    ${langRustJs}
  `;

  // Execute prettify in the context
  window.eval(scriptContent);

  // Run prettyPrint synchronously
  if (typeof window.prettyPrint === 'function') {
    // Find all code blocks and apply highlighting
    const codeBlocks = document.querySelectorAll('pre.lisp, pre.rust, pre.example');
    codeBlocks.forEach(block => {
      block.classList.add('prettyprint');
    });

    window.prettyPrint();
  }

  // Remove prettifier scripts from document
  const scripts = document.querySelectorAll('script.prettifier');
  scripts.forEach(script => script.remove());

  // Serialize back to HTML
  const doctype = document.doctype
    ? `<!DOCTYPE ${document.doctype.name}>\n`
    : '';
  const output = doctype + document.documentElement.outerHTML;

  fs.writeFileSync(filePath, output, 'utf8');

  window.close();
}

// Process all files sequentially
async function processAll() {
  for (const file of files) {
    try {
      await processFile(file);
    } catch (err) {
      console.error(`Failed to process ${file}:`, err.message);
    }
  }
}

processAll().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
