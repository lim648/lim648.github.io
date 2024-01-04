# TSMA_Tutorial

# Remove all objects from the current workspace to start with a clean environment
rm(list = ls())

# Load necessary libraries for the tutorial
library(caret)        # Load the caret package for data splitting, training, and performance evaluation
library(ROCR)         # Load the ROCR package for ROC analysis and performance metrics
library(tidyverse)    # Load the tidyverse package for data manipulation and visualization
library(mltools)      # Load the mltools package for machine learning tools and utilities



# Define a function 'metrics' to calculate various evaluation metrics
metrics <- function(pred.class, pred.prob, ref) {
  # pred.class: Predicted class labels
  # pred.prob: Predicted probabilities for the positive class
  # ref: Actual class labels (reference)
  
  # Calculate the confusion matrix using the caret package
  cm <- caret::confusionMatrix(data = pred.class, reference = ref, positive = '1')
  
  # Extract accuracy from the confusion matrix
  accuracy = cm$overall[['Accuracy']]
  
  # Extract precision from the confusion matrix
  precision = cm$byClass[['Precision']]
  
  # Extract recall from the confusion matrix
  recall = cm$byClass[['Recall']]
  
  # Extract specificity from the confusion matrix 
  specificity = cm$byClass[['Specificity']]
  
  # Calculate the balanced accuracy 
  balanced_acc = cm$byClass[['Balanced Accuracy']]
  
  # Calculate the F1 measure 
  f1_measure = cm$byClass[['F1']]
  
  # Create a prediction object for ROC analysis
  pred_model <- prediction(pred.prob, ref)
  
  # Calculatevthe Area Under the Precision-Recall Curve (AUCPR) 
  aucpr <- performance(pred_model, measure = "aucpr")@y.values[[1]]
  
  # Calculate Cohen's Kappa 
  kappa <- cm$overall[['Kappa']]
  
  # Calculate and round the Area Under the Receiver Operating Characteristic (AUROC) 
  auroc <- performance(pred_model, measure = "auc")@y.values[[1]]
  
  # Calculate Matthews Correlation Coefficient (MCC)
  mcc <- mcc(pred.class, ref)
  
  # Calculate the geometric mean of recall and specificity
  g_mean1 <- sqrt(recall * specificity)
  
  # Calculate the geometric mean of recall and precision
  g_mean2 <- sqrt(recall * precision)
  
  # Compile all the calculated metrics into a data frame
  output <- data.frame(accuracy = accuracy, precision = precision, recall = recall,
                       specificity = specificity, balanced_acc = balanced_acc, 
                       f1_measure = f1_measure, aucpr = aucpr, kappa = kappa,
                       auroc = auroc, mcc = mcc, g_mean1 = g_mean1, g_mean2 = g_mean2)
  
  # Round all numeric data in the data frame to 4 decimal places
  output <- round(output, 4)
  
  # Return the data frame with all metrics rounded to 4 digits
  return(output)
}

# Load the dataset from an RDS file
# This will read the 'unbalanced_disease' data into the environment
unbalanced_disease <- readRDS("~/Documents/camrsa/Data/unbalanced_disease.rds")

# Display the first 5 rows and 6 columns of the loaded dataset for a quick overview
unbalanced_disease[1:5, 1:7]

# Create a frequency table for the 'case_control' column to understand its distribution
table(unbalanced_disease$case_control)

# Data Splitting

# Set seed for reproducibility
set.seed(123)
# Create indices for a training set containing 80% of the data, ensuring the proportion of 'case_control' is maintained
train.rows <- createDataPartition(y = unbalanced_disease$case_control, p = 0.8, list = FALSE)

# Subset the dataset to create a training set using the indices, comprising 80% of the original data
unbalanced_disease_train <- unbalanced_disease[train.rows,]

# Create a testing set with the remaining 20% of the data not included in the training set
unbalanced_disease_test <- unbalanced_disease[-train.rows,]

# Model Training
############################## Elastic Net Alone ###############################
# Train an Elastic Net model
elastic <- train(
  case_control ~ ., data = unbalanced_disease_train, method = "glmnet",
  trControl = trainControl("cv", number = 5), # Set up cross-validation with 5 folds
  tuneLength = 10) # Specify the length of the tuning parameter grid

# Use the trained Elastic Net model to make predictions on the test dataset
predictions <- predict(elastic, unbalanced_disease_test)

# Predict the probabilities of the positive class for the test dataset using the Elastic Net model
predict <- predict(elastic, unbalanced_disease_test, type = "prob")[,2]

# Calculate various performance metrics (like accuracy, precision, recall, etc.) 
# for the Elastic Net model using the predicted classes and probabilities
el <- metrics(predictions, predict, unbalanced_disease_test$case_control)

############## Elastic Net with Under Resampling Techniques ####################

# Extract predictor variables (features) for the model by removing the 'case_control' column from the training dataset
x_un_el <- unbalanced_disease_train[,-which(names(unbalanced_disease_train) %in% c("case_control"))]

# Extract the target variable ('case_control') from the training dataset
y_un_el <- unbalanced_disease_train$case_control

# Apply under-sampling to balance the dataset, ensuring equal representation of each class in 'case_control'
under_caret_el <- downSample(x_un_el, y_un_el, yname = "case_control")

# Train an Elastic Net model using the balanced dataset
# - Specify the model formula where 'case_control' is the response variable and all other columns are predictors
# - Use 'glmnet' as the method for Elastic Net
# - Set cross-validation control using 'cv' with 5 folds for model validation
# - Define the length of the grid for hyperparameter tuning (tuneLength = 10)
elastic_under_el <- train(
  case_control ~ ., data = under_caret_el, method = "glmnet",
  trControl = trainControl("cv", number = 5),
  tuneLength = 10)

# Use the trained Elastic Net model with under-sampling to make predictions on the test set
predictions_el <- elastic_under_el %>% predict(unbalanced_disease_test)

# Predict the probabilities using the Elastic Net model and select the probabilities of the positive class
predict_el <- predict(elastic_under_el, unbalanced_disease_test, type = "prob")[,2]

# Evaluate the model using various metrics
un_el_downcaret <- metrics(predictions_el, predict_el, unbalanced_disease_test$case_control)

####################################### TSMA_EN ####################################
# Prepare for the Aggregation step
# Initialize a matrix to store the estimated probabilities for each bootstrap sample
estimated_probabilities_tsma <- matrix(NA, nrow = nrow(unbalanced_disease_test), ncol = 100)

# Iterate over 100 bootstrap samples
for(j in 1:100){
  # Stage 1: Bootstrap Samples
  # Create a bootstrap sample from the training dataset (90% of the training data)
  train_sample <- unbalanced_disease_train[sample(nrow(unbalanced_disease_train), 
                                                  size = round(0.90 * nrow(unbalanced_disease_train))), ]
  
  # Stage 2: Resampling
  # Extract features (excluding 'case_control') for resampling
  x_un <- train_sample[,-which(names(train_sample) %in% c("case_control"))]
  
  # Extract the target variable ('case_control')
  y_un <- train_sample$case_control
  
  # Apply under-sampling to balance the bootstrap sample
  under_caret <- downSample(x_un, y_un, yname = "case_control")
  
  # Model Training: Train an Elastic Net model on the downsampled data
  elastic_under_tsma <- train(
    case_control ~ ., data = under_caret, method = "glmnet",
    trControl = trainControl("cv", number = 5),
    tuneLength = 10)
  
  # Predict probabilities for the test set using the trained model
  predicted_tsma <- predict(elastic_under_tsma, unbalanced_disease_test, type = "prob")[,2]
  
  # Store the predicted probabilities in the matrix
  estimated_probabilities_tsma[, j] <- predicted_tsma
}

# Aggregation and Prediction
# Calculate the mean of the estimated probabilities for each observation
pred.prob_tsma <- rowMeans(estimated_probabilities_tsma, na.rm = TRUE)

# Convert the mean probabilities to class predictions based on a threshold of 0.5
pred.class_tsma <- as.factor(ifelse(pred.prob_tsma > 0.5, 1, 0))

# Evaluate the tsma model using various metrics
un_el_tsma <- metrics(pred.class_tsma, pred.prob_tsma, unbalanced_disease_test$case_control)



# Result
# Combine the results from the two models (Elastic Net with undersampling and TSMA-EN) into a single data frame
compare_result <- rbind(el, un_el_downcaret, un_el_tsma)

# Rename the columns of the resulting data frame for clarity
colnames(compare_result) = c("Accuracy", "Precision", "Recall", "Specificity", "Balanced Accuracy",
                             "F1 score", "AUC-PR", "Cohen's Kappa", "AUC-ROC", "MCC", "G Mean1", "G Mean2")

# Assign descriptive row names to identify each model's results
rownames(compare_result) = c("EN Alone", "EN with Under", "TSMA-EN")

# Display the comparison of performance metrics between the two models
compare_result



