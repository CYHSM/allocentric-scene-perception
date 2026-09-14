#!/usr/bin/env bash
# Everything the paper contains, from the result files, in one command.
#
#   bash scripts/build_paper.sh
#
# The order is not arbitrary: collate.py is the only script that reads a result
# file, and the tables and figures read what it writes. Run them out of order
# and nothing breaks -- they just report the previous collation, which is how a
# figure and a table come to disagree. So this exists, and nothing else should
# be run by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
if [ -x ".venv/bin/python3" ]; then
  PYTHON=".venv/bin/python3"
fi

echo "== collate =========================================================="
$PYTHON bench/collate.py

echo
echo "== tables ==========================================================="
$PYTHON bench/make_tables.py --latex | tee paper/tables.md

echo
echo "== figures =========================================================="
$PYTHON bench/make_figures.py

echo
echo "== prompts (appendix) ==============================================="
$PYTHON bench/make_prompts.py

echo
echo "paper/   runs.json observers.csv cells.csv trials.csv ladder.csv"
echo "         tables.md table1.tex table2.tex table3.tex prompts.tex"
echo "figures/ fig0, fig1, fig2, fig4 (png + pdf)"
