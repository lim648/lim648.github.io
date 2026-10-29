#!/usr/bin/env Rscript

# Reproducible TSMA tutorial analysis ---------------------------------------
#
# This script analyzes only the synthetic dataset created by
# R Code/generate_synthetic_data.R. It never reads restricted cohort data.

required_packages <- c("glmnet", "ROCR")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Install the required packages before running this tutorial: ",
    paste(missing_packages, collapse = ", ")
  )
}

analysis_seed <- 20261028L
test_fraction <- 0.20
n_bootstrap <- as.integer(Sys.getenv("TSMA_BOOTSTRAPS", "50"))
if (is.na(n_bootstrap) || n_bootstrap < 10L) {
  stop("TSMA_BOOTSTRAPS must be an integer of at least 10.")
}

data_path <- file.path("Data", "synthetic_unbalanced.rds")
results_dir <- "results"
results_path <- file.path(results_dir, "tutorial_metrics.csv")
metadata_path <- file.path(results_dir, "run_metadata.txt")

if (!file.exists(data_path)) {
  stop(
    "Synthetic data not found. Run `Rscript 'R Code/generate_synthetic_data.R'` first."
  )
}

safe_divide <- function(numerator, denominator) {
  if (is.na(denominator) || denominator == 0) return(NA_real_)
  numerator / denominator
}

stratified_split <- function(y, train_fraction, seed) {
  set.seed(seed)
  selected <- unlist(lapply(levels(y), function(level) {
    indices <- which(y == level)
    sample(indices, size = floor(train_fraction * length(indices)))
  }), use.names = FALSE)
  sort(selected)
}

downsample_training_data <- function(data, seed) {
  class_counts <- table(data$case_control)
  target_n <- min(class_counts)
  if (target_n < 5L) stop("Too few observations in a class for down-sampling.")

  set.seed(seed)
  selected <- unlist(lapply(levels(data$case_control), function(level) {
    sample(which(data$case_control == level), target_n, replace = FALSE)
  }), use.names = FALSE)
  data[sample(selected), , drop = FALSE]
}

make_model_matrix <- function(data) {
  predictors <- data[setdiff(names(data), "case_control")]
  stats::model.matrix(~ . - 1, data = predictors)
}

make_stratified_folds <- function(y, n_folds, seed) {
  set.seed(seed)
  fold_id <- integer(length(y))
  for (class_value in sort(unique(y))) {
    indices <- which(y == class_value)
    fold_id[indices] <- sample(rep(seq_len(n_folds), length.out = length(indices)))
  }
  fold_id
}

fit_elastic_net <- function(data, seed) {
  x <- make_model_matrix(data)
  y <- as.integer(data$case_control == "1")
  smallest_class <- min(table(y))
  n_folds <- min(5L, as.integer(smallest_class))
  if (n_folds < 3L) stop("At least three observations per class are required.")

  fit <- glmnet::cv.glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = 0.5,
    foldid = make_stratified_folds(y, n_folds, seed),
    type.measure = "deviance",
    standardize = TRUE
  )

  list(model = fit, feature_names = colnames(x))
}

predict_probabilities <- function(fit, new_data) {
  x <- make_model_matrix(new_data)
  missing_features <- setdiff(fit$feature_names, colnames(x))
  if (length(missing_features)) {
    padding <- matrix(
      0,
      nrow = nrow(x),
      ncol = length(missing_features),
      dimnames = list(NULL, missing_features)
    )
    x <- cbind(x, padding)
  }
  x <- x[, fit$feature_names, drop = FALSE]
  as.numeric(stats::predict(fit$model, newx = x, s = "lambda.min", type = "response"))
}

calculate_metrics <- function(actual, probability, threshold = 0.5) {
  actual <- factor(actual, levels = c("0", "1"))
  predicted <- factor(ifelse(probability >= threshold, "1", "0"), levels = c("0", "1"))

  tp <- sum(predicted == "1" & actual == "1")
  fp <- sum(predicted == "1" & actual == "0")
  tn <- sum(predicted == "0" & actual == "0")
  fn <- sum(predicted == "0" & actual == "1")

  accuracy <- safe_divide(tp + tn, tp + fp + tn + fn)
  precision <- safe_divide(tp, tp + fp)
  recall <- safe_divide(tp, tp + fn)
  specificity <- safe_divide(tn, tn + fp)
  balanced_accuracy <- mean(c(recall, specificity), na.rm = TRUE)
  f1 <- safe_divide(2 * precision * recall, precision + recall)
  g_mean1 <- sqrt(recall * specificity)
  g_mean2 <- sqrt(precision * recall)

  expected_agreement <- safe_divide(
    (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn),
    (tp + fp + tn + fn)^2
  )
  kappa <- safe_divide(accuracy - expected_agreement, 1 - expected_agreement)
  mcc <- safe_divide(
    tp * tn - fp * fn,
    sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  )

  rocr_prediction <- ROCR::prediction(probability, as.integer(actual == "1"))
  auroc <- ROCR::performance(rocr_prediction, "auc")@y.values[[1]]
  aucpr <- ROCR::performance(rocr_prediction, "aucpr")@y.values[[1]]

  data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    Balanced_Accuracy = balanced_accuracy,
    F1 = f1,
    AUC_PR = aucpr,
    Cohens_Kappa = kappa,
    AUC_ROC = auroc,
    MCC = mcc,
    G_Mean1 = g_mean1,
    G_Mean2 = g_mean2,
    stringsAsFactors = FALSE
  )
}

data <- readRDS(data_path)
required_columns <- c("case_control", "sex")
if (!all(required_columns %in% names(data))) {
  stop("The synthetic dataset is missing required columns.")
}
if (!identical(levels(data$case_control), c("0", "1"))) {
  stop("case_control must be a factor with levels c('0', '1').")
}
if (anyNA(data)) stop("The tutorial dataset must not contain missing values.")

train_indices <- stratified_split(
  data$case_control,
  train_fraction = 1 - test_fraction,
  seed = analysis_seed
)
train_data <- data[train_indices, , drop = FALSE]
test_data <- data[-train_indices, , drop = FALSE]

# 1. Elastic net trained on the original imbalanced training data.
fit_alone <- fit_elastic_net(train_data, analysis_seed + 1L)
prob_alone <- predict_probabilities(fit_alone, test_data)

# 2. Elastic net trained once after genuine random under-sampling.
under_data <- downsample_training_data(train_data, analysis_seed + 2L)
fit_under <- fit_elastic_net(under_data, analysis_seed + 3L)
prob_under <- predict_probabilities(fit_under, test_data)

# 3. TSMA: bootstrap with replacement, under-sample each bootstrap sample,
#    fit an elastic net, and average held-out probabilities.
tsma_probabilities <- matrix(
  NA_real_,
  nrow = nrow(test_data),
  ncol = n_bootstrap
)
failure_messages <- character()

for (bootstrap_id in seq_len(n_bootstrap)) {
  bootstrap_seed <- analysis_seed + 1000L + bootstrap_id
  set.seed(bootstrap_seed)
  bootstrap_indices <- sample(
    seq_len(nrow(train_data)),
    size = nrow(train_data),
    replace = TRUE
  )
  bootstrap_data <- train_data[bootstrap_indices, , drop = FALSE]

  result <- tryCatch({
    balanced_bootstrap <- downsample_training_data(
      bootstrap_data,
      seed = bootstrap_seed + 10000L
    )
    fit <- fit_elastic_net(balanced_bootstrap, bootstrap_seed + 20000L)
    predict_probabilities(fit, test_data)
  }, error = function(error) {
    failure_messages <<- c(
      failure_messages,
      sprintf("bootstrap %d: %s", bootstrap_id, conditionMessage(error))
    )
    rep(NA_real_, nrow(test_data))
  })

  tsma_probabilities[, bootstrap_id] <- result
}

successful_bootstraps <- sum(colSums(is.finite(tsma_probabilities)) == nrow(test_data))
minimum_successes <- max(10L, ceiling(0.8 * n_bootstrap))
if (successful_bootstraps < minimum_successes) {
  stop(
    "Only ", successful_bootstraps, " of ", n_bootstrap,
    " bootstrap models succeeded; at least ", minimum_successes, " are required."
  )
}
prob_tsma <- rowMeans(tsma_probabilities, na.rm = TRUE)

results <- rbind(
  cbind(Method = "EN alone", calculate_metrics(test_data$case_control, prob_alone)),
  cbind(Method = "EN with UNDER", calculate_metrics(test_data$case_control, prob_under)),
  cbind(Method = "TSMA-EN", calculate_metrics(test_data$case_control, prob_tsma))
)

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
utils::write.csv(results, results_path, row.names = FALSE, na = "NA")

metadata <- c(
  "This analysis uses a fully synthetic dataset and no restricted cohort records.",
  sprintf("Analysis seed: %d", analysis_seed),
  sprintf("Rows: %d", nrow(data)),
  sprintf("Predictors: %d", ncol(data) - 1L),
  sprintf("Observed positive prevalence: %.4f", mean(data$case_control == "1")),
  sprintf("Train rows: %d", nrow(train_data)),
  sprintf("Test rows: %d", nrow(test_data)),
  sprintf("Requested TSMA bootstrap models: %d", n_bootstrap),
  sprintf("Successful TSMA bootstrap models: %d", successful_bootstraps),
  sprintf("Failed TSMA bootstrap models: %d", length(failure_messages)),
  sprintf("R version: %s", R.version.string),
  sprintf("glmnet version: %s", as.character(utils::packageVersion("glmnet"))),
  sprintf("ROCR version: %s", as.character(utils::packageVersion("ROCR"))),
  sprintf("Synthetic data MD5: %s", unname(tools::md5sum(data_path)))
)
if (length(failure_messages)) {
  metadata <- c(metadata, "Failure messages:", failure_messages)
}
writeLines(metadata, metadata_path)

print_data <- results
numeric_columns <- vapply(print_data, is.numeric, logical(1))
print_data[numeric_columns] <- lapply(print_data[numeric_columns], round, digits = 4)
print(print_data, row.names = FALSE)
cat("\nResults written to", results_path, "\n")
cat("Run metadata written to", metadata_path, "\n")
