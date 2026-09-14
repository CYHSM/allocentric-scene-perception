#!/usr/bin/env bash
# Compile the paper. Tables and figures come from `bash scripts/build_paper.sh`
# and are \input / \includegraphics'd from ../ and ../../figures -- so refresh
# those first if any run has landed since the last compile.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
grep -c '^!' main.log && echo "LaTeX errors above" || true
echo "main.pdf: $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages"
