#!/usr/bin/env Rscript

# Evaluate the synthetic scenario across prespecified seeds before choosing a
# representative dataset. The representative is closest to the median effect,
# not the seed with the largest TSMA improvement.

source(file.path("R Code", "generate_synthetic_data.R"))
source(file.path("R Code", "tsma_core.R"))

benchmark_seeds <- 20261101:20261112
n_bootstrap <- as.integer(Sys.getenv("TSMA_BENCHMARK_BOOTSTRAPS", "30"))
if (is.na(n_bootstrap) || n_bootstrap < 10L) {
  stop("TSMA_BENCHMARK_BOOTSTRAPS must be an integer of at least 10.")
}

results_dir <- "results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

evaluate_seed <- function(generation_seed) {
  message("Benchmarking generation seed ", generation_seed)
  data <- generate_synthetic_dataset(generation_seed)
  evaluation <- evaluate_tsma_methods(
    data = data,
    analysis_seed = generation_seed + 50000L,
    n_bootstrap = n_bootstrap
  )
  cbind(Generation_Seed = generation_seed, evaluation$results)
}

benchmark_results <- do.call(rbind, lapply(benchmark_seeds, evaluate_seed))
utils::write.csv(
  benchmark_results,
  file.path(results_dir, "seed_benchmark.csv"),
  row.names = FALSE,
  na = "NA"
)

under_results <- benchmark_results[
  benchmark_results$Method == "EN with UNDER",
  ,
  drop = FALSE
]
tsma_results <- benchmark_results[
  benchmark_results$Method == "TSMA-EN",
  ,
  drop = FALSE
]
stopifnot(identical(under_results$Generation_Seed, tsma_results$Generation_Seed))

metric_names <- c("F1", "Balanced_Accuracy", "AUC_PR", "AUC_ROC")
delta_table <- data.frame(Generation_Seed = under_results$Generation_Seed)
for (metric_name in metric_names) {
  delta_table[[metric_name]] <-
    tsma_results[[metric_name]] - under_results[[metric_name]]
}
utils::write.csv(
  delta_table,
  file.path(results_dir, "tsma_minus_under_by_seed.csv"),
  row.names = FALSE
)

benchmark_summary <- do.call(rbind, lapply(metric_names, function(metric_name) {
  values <- delta_table[[metric_name]]
  data.frame(
    Metric = metric_name,
    Mean_Delta = mean(values),
    Median_Delta = median(values),
    SD_Delta = stats::sd(values),
    Positive_Repetitions = sum(values > 0),
    Total_Repetitions = length(values),
    stringsAsFactors = FALSE
  )
}))
utils::write.csv(
  benchmark_summary,
  file.path(results_dir, "seed_benchmark_summary.csv"),
  row.names = FALSE
)

# F1 and AUC-PR are prespecified as the primary rare-class metrics. Select the
# seed whose two deltas are jointly closest to their medians. This explicitly
# prevents choosing the seed with the most favorable observed improvement.
primary_metrics <- c("F1", "AUC_PR")
distance_components <- vapply(primary_metrics, function(metric_name) {
  values <- delta_table[[metric_name]]
  scale_value <- stats::mad(values, constant = 1)
  if (!is.finite(scale_value) || scale_value == 0) scale_value <- 1
  abs(values - median(values)) / scale_value
}, numeric(nrow(delta_table)))
representative_score <- rowSums(distance_components)
representative_seed <- min(
  delta_table$Generation_Seed[
    representative_score == min(representative_score)
  ]
)

writeLines(
  as.character(representative_seed),
  file.path(results_dir, "representative_seed.txt")
)
write_synthetic_dataset(representative_seed)

benchmark_metadata <- c(
  "Prespecified benchmark seeds: 20261101 through 20261112 inclusive.",
  sprintf("Benchmark TSMA bootstrap models per seed: %d", n_bootstrap),
  "Primary seed-selection metrics: F1 and AUC-PR.",
  "Selection rule: minimum standardized absolute distance from the median TSMA-minus-UNDER deltas.",
  sprintf("Representative generation seed: %d", representative_seed),
  "No seed was removed after results were observed."
)
writeLines(
  benchmark_metadata,
  file.path(results_dir, "benchmark_metadata.txt")
)

print(benchmark_summary, row.names = FALSE)
cat("Representative generation seed:", representative_seed, "\n")
