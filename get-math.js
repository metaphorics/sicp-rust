#!/usr/bin/env node

// Usage: ./get-math.js json_db file1 [file2 ...]
//  json_db: the JSON file that will contain the database
//    (existing file will be overwritten if math has changed in the source);
//  file1, file2, ...: the input files to search for LaTeX strings.

// This extracts LaTeX from given files and converts it to MathML using
// MathJax. It then builds up a database of key/value pairs mapping from
// LaTeX to MathML. The result is a JSON object that is written to json_db.
// LaTeX must be delimited by \( \) (inline) or \[ \] (display math).

// (c) 2014 Andres Raba, GNU GPL v.3.
// (c) 2025 Updated for Node.js with mathjax-node

const fs = require("fs");
const mjAPI = require("mathjax-node");

// Configure MathJax
mjAPI.config({
  MathJax: {
    // MathJax configuration
  }
});
mjAPI.start();

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

// Convert LaTeX to MathML using mathjax-node
async function convertLatex(latex) {
  // Determine if inline or display math
  const isInline = latex.startsWith("\\(");

  // Strip delimiters for MathJax
  let math;
  if (isInline) {
    math = latex.slice(2, -2); // Remove \( and \)
  } else {
    math = latex.slice(2, -2); // Remove \[ and \]
  }

  return new Promise((resolve, reject) => {
    mjAPI.typeset({
      math: math,
      format: isInline ? "inline-TeX" : "TeX",
      mml: true,
    }, data => {
      if (data.errors) {
        console.error(`MathJax error for "${latex}":`, data.errors);
        resolve(""); // Return empty on error
      } else {
        resolve(data.mml);
      }
    });
  });
}

async function main() {
  const deltaKeys = Object.keys(delta);

  if (deltaKeys.length === 0) {
    // No new LaTeX to convert, just write existing
    const jsonObj = JSON.stringify(texobj, null, 2);
    fs.writeFileSync(db, jsonObj, "utf8");
    process.exit(0);
  }

  // Convert all new LaTeX
  for (const latex of deltaKeys) {
    const mml = await convertLatex(latex);
    texobj[latex] = mml;
  }

  // Write the database
  const jsonObj = JSON.stringify(texobj, null, 2);
  fs.writeFileSync(db, jsonObj, "utf8");
}

main().catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
