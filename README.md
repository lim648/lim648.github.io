# Reproducible TSMA tutorial for imbalanced data in R

This repository contains a reproducible tutorial for two-stage sampling aggregation (TSMA). The current example uses **fully synthetic data only**. It contains no EMerge records or values derived from restricted individual-level data.

## Reproduce everything

Install R packages `glmnet`, `ROCR`, `knitr`, and `rmarkdown`, plus [Quarto](https://quarto.org/). Then run from the repository root:

```bash
Rscript "R Code/render_tutorial.R"
```

That command regenerates:

- `Data/synthetic_unbalanced.rds` from a fixed seed;
- `results/tutorial_metrics.csv` and `results/run_metadata.txt`;
- `TSMA_Tutorial.html` from `TSMA_Tutorial.qmd`.

View the rendered tutorial at [lim648.github.io/TSMA_Tutorial.html](https://lim648.github.io/TSMA_Tutorial.html).

## Reproducibility and interpretation

The result table is generated directly by the published analysis code and must not be edited by hand. The synthetic example demonstrates implementation; it does not guarantee that TSMA will outperform every comparator on every metric or dataset.

Restricted cohort data must never be committed to this repository, even after column renaming or row subsetting.
