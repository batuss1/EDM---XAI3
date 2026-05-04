# XAI Exercise 5: Partial Dependence Plots

This repository contains the solution for the XAI model-agnostic PDP exercise.

## Contents

- `data/`: bike rental and house price datasets used in the analysis.
- `scripts/pdp_analysis.py`: reproducible Python script that trains the random forest models and generates PDP figures.
- `reports/xai3_report.tex`: LaTeX source for the report.
- `reports/xai3_report.pdf`: final report with comments and answers.
- `reports/figures/`: generated PDP plots.

## Reproduce

Install the Python dependencies and run:

```powershell
python scripts/pdp_analysis.py
```

Then compile the report from the `reports/` folder:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error xai3_report.tex
```

