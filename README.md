# Reproducible TSMA tutorial for imbalanced data in R

This repository contains a reproducible tutorial for two-stage sampling aggregation (TSMA). The current example uses **fully synthetic data only**. It contains no EMerge records or values derived from restricted individual-level data.

## Reproduce everything

Install R packages `glmnet`, `ROCR`, `knitr`, and `rmarkdown`, plus [Quarto](https://quarto.org/). Then run from the repository root:

```bash
Rscript "R Code/render_tutorial.R"
```

That command regenerates:

- a 12-seed benchmark defined before model fitting;
- `Data/synthetic_unbalanced.rds` from the median-effect representative seed;
- `results/tutorial_metrics.csv` and `results/run_metadata.txt`;
- the per-seed results, TSMA-minus-UNDER summary, and seed-selection record;
- `TSMA_Tutorial.html` from `TSMA_Tutorial.qmd`.

View the rendered tutorial at [lim648.github.io/TSMA_Tutorial.html](https://lim648.github.io/TSMA_Tutorial.html).

## Reproducibility and interpretation

The result tables are generated directly by the published analysis code and must not be edited by hand. The synthetic scenario is designed to test TSMA's stated advantage over one random under-sample: lower dependence on a single small balanced sample. All 12 prespecified seeds are retained, and the displayed dataset is closest to the median F1/AUC-PR effect rather than the maximum effect. The example does not imply that TSMA will outperform every comparator on every metric or dataset.

Restricted cohort data must never be committed to this repository, even after column renaming or row subsetting.
