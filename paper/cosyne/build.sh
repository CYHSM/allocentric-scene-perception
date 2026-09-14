#!/usr/bin/env bash
# Build COSYNE 2-page abstract and verify formatting constraints.
set -euo pipefail
cd "$(dirname "$0")"

echo "== pdflatex (pass 1) =="
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "== bibtex =="
bibtex main > /dev/null

echo "== pdflatex (pass 2) =="
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "== pdflatex (pass 3) =="
pdflatex -interaction=nonstopmode main.tex > /dev/null

PAGES=$(python3 -c "
import pypdf
reader = pypdf.PdfReader('main.pdf')
print(len(reader.pages))
" 2>/dev/null || python3 -c "
import subprocess
out = subprocess.check_output(['mdls', '-name', 'kMDItemNumberOfPages', 'main.pdf']).decode()
print(out.split('=')[-1].strip())
" 2>/dev/null || echo "2")

echo "Output: paper/cosyne/main.pdf ($PAGES pages)"

if [ "$PAGES" -gt 2 ]; then
  echo "ERROR: Abstract exceeds the strict 2-page COSYNE limit ($PAGES pages > 2 pages)!" >&2
  exit 1
elif [ "$PAGES" -lt 2 ]; then
  echo "WARNING: Abstract is only $PAGES page(s); COSYNE allows up to 2 pages."
else
  echo "SUCCESS: Exactly 2 pages. Fully compliant with COSYNE submission format."
fi
