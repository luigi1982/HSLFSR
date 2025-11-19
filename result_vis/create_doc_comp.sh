#!/usr/bin/env bash
set -e

NAME=$1

source .venv/bin/activate

python result_vis/compare_results.py --name $NAME
pdflatex result_vis/latex/compare_$NAME.tex
mv compare_$NAME.pdf reports

rm compare_$NAME.aux
rm compare_$NAME.log
rm compare_$NAME.toc