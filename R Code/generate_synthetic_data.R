#!/usr/bin/env Rscript

# Generate a fully synthetic, imbalanced binary-classification dataset.
# No row or value is derived from EMerge or any other restricted cohort.

generation_seed <- 20261028L
n_observations <- 2500L
n_features <- 30L
target_prevalence <- 0.05
output_path <- file.path("Data", "synthetic_unbalanced.rds")

set.seed(generation_seed)

# Correlated continuous predictors with five latent components.
latent <- matrix(rnorm(n_observations * 5L), ncol = 5L)
features <- vapply(seq_len(n_features), function(feature_id) {
  component <- 1L + ((feature_id - 1L) %% ncol(latent))
  0.55 * latent[, component] + sqrt(1 - 0.55^2) * rnorm(n_observations)
}, numeric(n_observations))
colnames(features) <- sprintf("feature_%03d", seq_len(n_features))

sex <- factor(
  sample(c("female", "male"), n_observations, replace = TRUE),
  levels = c("female", "male")
)

linear_predictor <-
  0.85 * features[, 1] -
  0.70 * features[, 2] +
  0.55 * features[, 3] +
  0.45 * features[, 4] * features[, 5] +
  0.30 * (sex == "male")

intercept <- uniroot(
  function(value) mean(plogis(value + linear_predictor)) - target_prevalence,
  interval = c(-12, 2)
)$root
event_probability <- plogis(intercept + linear_predictor)
case_control <- factor(
  rbinom(n_observations, size = 1L, prob = event_probability),
  levels = c(0, 1),
  labels = c("0", "1")
)

synthetic_data <- data.frame(
  case_control = case_control,
  sex = sex,
  features,
  check.names = FALSE
)

dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
saveRDS(synthetic_data, output_path, version = 3)

cat("Synthetic tutorial data written to", output_path, "\n")
cat("Rows:", nrow(synthetic_data), "\n")
cat("Predictors:", ncol(synthetic_data) - 1L, "\n")
cat("Observed positive prevalence:", round(mean(case_control == "1"), 4), "\n")
