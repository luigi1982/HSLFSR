source .venv/bin/activate

python result_vis/master_comp_vis.py
pdflatex result_vis/master/v1.tex

rm v1.aux
rm v1.log
rm v1.log