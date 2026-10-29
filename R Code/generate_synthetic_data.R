#!/usr/bin/env Rscript

# Generate a fully synthetic, imbalanced binary-classification dataset.
# No row or value is derived from EMerge or any other restricted cohort.

generate_synthetic_dataset <- function(
  generation_seed,
  n_observations = 3000L,
  n_features = 120L,
  target_prevalence = 0.03
) {
  stopifnot(
    length(generation_seed) == 1L,
    n_observations > 100L,
    n_features >= 8L,
    target_prevalence > 0,
    target_prevalence < 0.5
  )

  set.seed(as.integer(generation_seed))

  # Moderate correlation makes the sparse, high-dimensional problem realistic
  # while keeping the true signal fully documented and reproducible.
  latent <- matrix(rnorm(n_observations * 8L), ncol = 8L)
  features <- vapply(seq_len(n_features), function(feature_id) {
    component <- 1L + ((feature_id - 1L) %% ncol(latent))
    0.30 * latent[, component] + sqrt(1 - 0.30^2) * rnorm(n_observations)
  }, numeric(n_observations))
  colnames(features) <- sprintf("feature_%03d", seq_len(n_features))

  signal_coefficients <- c(1.25, -1.10, 0.95, -0.85, 0.75, -0.65, 0.55, -0.45)
  linear_predictor <- drop(
    features[, seq_along(signal_coefficients), drop = FALSE] %*%
      signal_coefficients
  )

  intercept <- uniroot(
    function(value) {
      mean(plogis(value + linear_predictor)) - target_prevalence
    },
    interval = c(-15, 2)
  )$root

  event_probability <- plogis(intercept + linear_predictor)
  case_control <- factor(
    rbinom(n_observations, size = 1L, prob = event_probability),
    levels = c(0, 1),
    labels = c("0", "1")
  )

  data.frame(
    case_control = case_control,
    features,
    check.names = FALSE
  )
}

write_synthetic_dataset <- function(
  generation_seed,
  output_path = file.path("Data", "synthetic_unbalanced.rds")
) {
  synthetic_data <- generate_synthetic_dataset(generation_seed)
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(synthetic_data, output_path, version = 3)

  cat("Synthetic tutorial data written to", output_path, "\n")
  cat("Generation seed:", generation_seed, "\n")
  cat("Rows:", nrow(synthetic_data), "\n")
  cat("Predictors:", ncol(synthetic_data) - 1L, "\n")
  cat(
    "Observed positive prevalence:",
    round(mean(synthetic_data$case_control == "1"), 4),
    "\n"
  )
  invisible(synthetic_data)
}

if (sys.nframe() == 0L) {
  seed_path <- file.path("results", "representative_seed.txt")
  generation_seed <- if (file.exists(seed_path)) {
    as.integer(readLines(seed_path, n = 1L))
  } else {
    20261101L
  }
  write_synthetic_dataset(generation_seed)
}
