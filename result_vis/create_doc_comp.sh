#!/usr/bin/env bash
set -e

source .venv/bin/activate

python result_vis/compare_results.py
pdflatex result_vis/latex/compare.tex
mv compare.pdf reports

rm compare.aux
rm compare.log
rm compare.toc