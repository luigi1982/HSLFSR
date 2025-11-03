#!/usr/bin/env bash
set -e

MODEL=$1
EXP=$2

FILE=result_vis/latex/$MODEL/$MODEL-$EXP.tex
DIR=reports/$MODEL

source .venv/bin/activate

python result_vis/generate_vis.py --model $MODEL --exp $EXP
pdflatex $FILE
mkdir -p $DIR || true
mv $MODEL-$EXP.pdf $DIR

rm $MODEL-$EXP.aux
rm $MODEL-$EXP.log
rm texput.log