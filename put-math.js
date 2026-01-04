#!/usr/bin/env node

// Usage: ./put-math.js json_db file1 [file2 ...]
//  json_db: existing JSON file that contains the MathML database;
//  file1, file2, ...: the output files to modify.

// This replaces all the LaTeX markup in given output files with MathML.
// It does it by searching for LaTeX strings delimited by \( \) or \[ \]
// and looks up the mapping from LaTeX to MathML in the JSON database.

// (c) 2014 Andres Raba, GNU GPL v.3.
// (c) 2025 Updated for Node.js

const fs = require("fs");

// LaTeX is enclosed in \( \) or \[ \] delimiters,
// first pair for inline, second for display math:
const pattern = /\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]/g;

// Get command-line arguments
const args = process.argv.slice(2);

if (args.length <= 1) {
  console.log("Usage: ./put-math.js json_db file1 [file2 ...]");
  process.exit(0);
}

const db = args[0]; // JSON database
const inputFiles = args.slice(1); // file1, file2, ...

// Load the MathML database
let mathml;
try {
  mathml = JSON.parse(fs.readFileSync(db, "utf8"));
} catch (error) {
  console.error("Error reading database:", error.message);
  process.exit(1);
}

// Process each file
inputFiles.forEach(arg => {
  try {
    let file = fs.readFileSync(arg, "utf8");
    // Replace LaTeX with MathML or paint LaTeX blue
    // if mapping not found in JSON database:
    file = file.replace(pattern, latex => {
      return mathml[latex] || "<span style='color:blue'>" + latex + "</span>";
    });
    fs.writeFileSync(arg, file, "utf8");
  } catch (error) {
    console.error(`Error processing ${arg}:`, error.message);
  }
});
