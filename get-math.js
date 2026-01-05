#!/usr/bin/env node

// Usage: ./get-math.js json_db file1 [file2 ...]
//  json_db: the JSON file that will contain the database
//    (existing file will be overwritten if math has changed in the source);
//  file1, file2, ...: the input files to search for LaTeX strings.

// This extracts LaTeX from given files and converts it to MathML using
// MathJax v3 (mathjax-full). It then builds up a database of key/value pairs
// mapping from LaTeX to MathML. The result is a JSON object that is written
// to json_db. LaTeX must be delimited by \( \) (inline) or \[ \] (display math).

// (c) 2014 Andres Raba, GNU GPL v.3.
// (c) 2025 Updated for MathJax v3 (mathjax-full)

const fs = require("fs");

// MathJax v3 imports
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { SerializedMmlVisitor } = require("mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js");
const { STATE } = require("mathjax-full/js/core/MathItem.js");

// Import TeX packages
require("mathjax-full/js/input/tex/base/BaseConfiguration.js");
require("mathjax-full/js/input/tex/ams/AmsConfiguration.js");
require("mathjax-full/js/input/tex/newcommand/NewcommandConfiguration.js");
require("mathjax-full/js/input/tex/noundefined/NoUndefinedConfiguration.js");

// Setup MathJax
const EM = 16;
const EX = 8;
const WIDTH = 80 * EM;

const adaptor = liteAdaptor({ fontSize: EM });
RegisterHTMLHandler(adaptor);

const tex = new TeX({
  packages: ["base", "ams", "newcommand", "noundefined"],
  formatError(jax, err) {
    console.error("TeX error:", err.message);
    return null;
  },
});

const html = mathjax.document("", { InputJax: tex });

// Create MathML serializer
const visitor = new SerializedMmlVisitor();
const toMathML = (node) => visitor.visitTree(node, html);

// LaTeX is enclosed in \( \) or \[ \] delimiters,
// first pair for inline, second for display math:
const pattern = /\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]/g;

// Get command-line arguments
const args = process.argv.slice(2);

if (args.length <= 1) {
  console.log("Usage: ./get-math.js json_db file1 [file2 ...]");
  process.exit(0);
}

const db = args[0];
const inputFiles = args.slice(1);

// Will hold extracted LaTeX fragments as keys:
const texobj = {};

// Extract LaTeX from all input files
inputFiles.forEach(arg => {
  try {
    const file = fs.readFileSync(arg, "utf8");
    let matched;
    while ((matched = pattern.exec(file)) != null) {
      texobj[matched[0]] = "";
    }
  } catch (error) {
    console.error(`Error reading ${arg}:`, error.message);
    process.exit(1);
  }
});

// Load existing database if available
let oldmath = {};
if (fs.existsSync(db)) {
  try {
    oldmath = JSON.parse(fs.readFileSync(db, "utf8"));
  } catch (error) {
    console.error("Error reading existing database:", error.message);
  }
}

// Collect only LaTeX that needs conversion (not in old database)
const delta = {};
Object.keys(texobj).forEach(latex => {
  if (oldmath[latex]) {
    texobj[latex] = oldmath[latex];
  } else {
    delta[latex] = "";
  }
});

// Convert LaTeX to MathML using MathJax v3
function convertLatex(latex) {
  // Determine if inline or display math
  const isInline = latex.startsWith("\\(");

  // Strip delimiters for MathJax
  let math;
  if (isInline) {
    math = latex.slice(2, -2); // Remove \( and \)
  } else {
    math = latex.slice(2, -2); // Remove \[ and \]
  }

  try {
    const node = html.convert(math, {
      display: !isInline,
      em: EM,
      ex: EX,
      containerWidth: WIDTH,
      end: STATE.CONVERT,
    });
    const mml = toMathML(node);
    return mml;
  } catch (error) {
    console.error(`MathJax error for "${latex.substring(0, 50)}...":`, error.message);
    return ""; // Return empty on error
  }
}

function main() {
  const deltaKeys = Object.keys(delta);

  if (deltaKeys.length === 0) {
    console.log("No new LaTeX to convert.");
    // No new LaTeX to convert, just write existing
    const jsonObj = JSON.stringify(texobj, null, 2);
    fs.writeFileSync(db, jsonObj, "utf8");
    process.exit(0);
  }

  console.log(`Converting ${deltaKeys.length} LaTeX expressions to MathML...`);

  // Convert all new LaTeX
  let success = 0;
  let failed = 0;
  for (const latex of deltaKeys) {
    const mml = convertLatex(latex);
    if (mml) {
      texobj[latex] = mml;
      success++;
    } else {
      failed++;
    }
  }

  // Write the database
  const jsonObj = JSON.stringify(texobj, null, 2);
  fs.writeFileSync(db, jsonObj, "utf8");
  console.log(`Wrote ${Object.keys(texobj).length} entries to ${db} (${success} converted, ${failed} failed)`);
}

main();
