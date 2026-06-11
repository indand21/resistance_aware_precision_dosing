"""
convert_md_to_docx.py
=====================
Convert a markdown manuscript to a formatted Word (.docx) document via pandoc.

Handles the features this project's manuscript uses:
  * YAML frontmatter (title/authors/keywords -> docx metadata)
  * LaTeX math ($...$ inline and $$...$$ display -> native Office Math / OMML)
  * pipe tables
  * image references written relative to the manuscript file (e.g.
    ``../figures/foo.png``): pandoc's --resource-path is pointed at the
    markdown file's own directory so those relative paths resolve and the
    figures are embedded.

Usage:
    python convert_md_to_docx.py INPUT.md [OUTPUT.docx] [--reference-doc REF.docx]

If OUTPUT is omitted it defaults to INPUT with a .docx extension.
Requires pandoc on PATH (tested with pandoc 3.8).
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys


def convert(input_md: str, output_docx: str | None = None,
            reference_doc: str | None = None) -> str:
    input_md = os.path.abspath(input_md)
    if not os.path.isfile(input_md):
        raise FileNotFoundError(f"Input markdown not found: {input_md}")
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found on PATH. Install pandoc (https://pandoc.org).")

    if output_docx is None or output_docx == "":
        output_docx = os.path.splitext(input_md)[0] + ".docx"
    output_docx = os.path.abspath(output_docx)

    md_dir = os.path.dirname(input_md)

    # Pandoc reads GitHub-flavoured-ish markdown with TeX math and pipe tables.
    cmd = [
        "pandoc",
        input_md,
        "-f", "markdown+tex_math_dollars+pipe_tables+yaml_metadata_block",
        "-t", "docx",
        "-o", output_docx,
        # Resolve image paths written relative to the markdown file (../figures/...)
        # against the markdown file's directory.
        "--resource-path", os.pathsep.join([md_dir, os.getcwd()]),
        "--standalone",
    ]
    if reference_doc:
        cmd += ["--reference-doc", os.path.abspath(reference_doc)]

    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"pandoc failed (exit {proc.returncode}).")
    if proc.stderr.strip():
        # pandoc warnings (e.g. a missing image) go to stderr but are non-fatal.
        print("pandoc warnings:\n" + proc.stderr.strip())

    size = os.path.getsize(output_docx)
    print(f"\nInput : {input_md}")
    print(f"Output: {output_docx}")
    print(f"Size  : {size:,} bytes")
    if size == 0:
        raise RuntimeError("Output file is empty.")
    return output_docx


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert a markdown manuscript to .docx via pandoc.")
    p.add_argument("input", help="input markdown file")
    p.add_argument("output", nargs="?", default=None, help="output .docx (default: input with .docx)")
    p.add_argument("--reference-doc", default=None,
                   help="optional reference .docx providing Word styles")
    args = p.parse_args(argv)
    convert(args.input, args.output, args.reference_doc)


if __name__ == "__main__":
    main()
