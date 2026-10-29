# Shared model-fitting and evaluation functions for the reproducible tutorial.

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

safe_divide <- function(numerator, denominator, zero_value = NA_real_) {
  if (is.na(denominator) || denominator == 0) return(zero_value)
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
    type.measure = "auc",
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
  as.numeric(
    stats::predict(fit$model, newx = x, s = "lambda.1se", type = "response")
  )
}

calculate_metrics <- function(actual, probability, threshold = 0.5) {
  actual <- factor(actual, levels = c("0", "1"))
  predicted <- factor(
    ifelse(probability >= threshold, "1", "0"),
    levels = c("0", "1")
  )

  tp <- sum(predicted == "1" & actual == "1")
  fp <- sum(predicted == "1" & actual == "0")
  tn <- sum(predicted == "0" & actual == "0")
  fn <- sum(predicted == "0" & actual == "1")

  accuracy <- safe_divide(tp + tn, tp + fp + tn + fn)
  # Defining precision and F1 as zero when no positives are predicted avoids
  # reporting an apparently missing result for a genuine model failure mode.
  precision <- safe_divide(tp, tp + fp, zero_value = 0)
  recall <- safe_divide(tp, tp + fn, zero_value = 0)
  specificity <- safe_divide(tn, tn + fp, zero_value = 0)
  balanced_accuracy <- mean(c(recall, specificity))
  f1 <- safe_divide(2 * precision * recall, precision + recall, zero_value = 0)
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

evaluate_tsma_methods <- function(
  data,
  analysis_seed,
  n_bootstrap = 50L,
  test_fraction = 0.20
) {
  if (!identical(levels(data$case_control), c("0", "1"))) {
    stop("case_control must be a factor with levels c('0', '1').")
  }
  if (anyNA(data)) stop("The tutorial dataset must not contain missing values.")
  if (n_bootstrap < 10L) stop("At least 10 bootstrap models are required.")

  train_indices <- stratified_split(
    data$case_control,
    train_fraction = 1 - test_fraction,
    seed = analysis_seed
  )
  train_data <- data[train_indices, , drop = FALSE]
  test_data <- data[-train_indices, , drop = FALSE]

  fit_alone <- fit_elastic_net(train_data, analysis_seed + 1L)
  prob_alone <- predict_probabilities(fit_alone, test_data)

  under_data <- downsample_training_data(train_data, analysis_seed + 2L)
  fit_under <- fit_elastic_net(under_data, analysis_seed + 3L)
  prob_under <- predict_probabilities(fit_under, test_data)

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

  successful_bootstraps <- sum(
    colSums(is.finite(tsma_probabilities)) == nrow(test_data)
  )
  minimum_successes <- max(10L, ceiling(0.8 * n_bootstrap))
  if (successful_bootstraps < minimum_successes) {
    stop(
      "Only ", successful_bootstraps, " of ", n_bootstrap,
      " bootstrap models succeeded; at least ", minimum_successes,
      " are required."
    )
  }
  prob_tsma <- rowMeans(tsma_probabilities, na.rm = TRUE)

  results <- rbind(
    cbind(Method = "EN alone", calculate_metrics(test_data$case_control, prob_alone)),
    cbind(
      Method = "EN with UNDER",
      calculate_metrics(test_data$case_control, prob_under)
    ),
    cbind(Method = "TSMA-EN", calculate_metrics(test_data$case_control, prob_tsma))
  )

  list(
    results = results,
    train_data = train_data,
    test_data = test_data,
    successful_bootstraps = successful_bootstraps,
    failure_messages = failure_messages
  )
}
