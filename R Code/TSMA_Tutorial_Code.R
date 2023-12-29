# TSMA_Tutorial

rm(list=ls())
library(randomForest)
library(caret)
library(ROCR)
library(tidyverse)
library(mltools)

# Calculate all the performance metrics
metrics <- function(pred.class, pred.prob, ref){
  met <-c(NA,12)
  
  cm <- caret::confusionMatrix(data= pred.class, reference = ref,positive='1')
  accuracy = round(cm$overall[['Accuracy']], 4); met[1] = accuracy
  precision = round(cm$byClass[['Precision']], 4); met[2] = precision
  recall = round(cm$byClass[['Recall']], 4); met[3] = recall
  specificity= round(cm$byClass[['Specificity']], 4) ; met[4] = specificity
  balanced_acc= round(cm$byClass[['Balanced Accuracy']],4); met[5] = balanced_acc
  f1_measure = round(cm$byClass[['F1']],4); met[6] = f1_measure
  f2_measure = ((1+2^2)*(cm$byClass[['Precision']])*(cm$byClass[['Recall']]))/(2^2*(cm$byClass[['Precision']])+(cm$byClass[['Recall']]))
  f2_measure<-round(f2_measure,4);met[7] = f2_measure
  pred_model<-prediction(pred.prob,ref)
  AUCPR<-round(performance(pred_model, measure = "aucpr")@y.values[[1]], 4); met[8] = AUCPR
  Kappa<-round(cm$overall[['Kappa']], 4); met[9] = Kappa
  AUROC <- round(performance(pred_model, measure = "auc")@y.values[[1]], 4); met[10] = AUROC
  MCC<-round(mcc(pred.class,ref),4); met[11] = MCC
  G_mean<-round(sqrt(precision *specificity),4);met[12]=G_mean
  
  return(met)
}

# Loading the Data
unbalanced_disease <- readRDS("~/Documents/camrsa/Data/unbalanced_disease.rds")
unbalanced_disease[1:5,1:6]
table(unbalanced_disease$CASE_CONTROL)

# Data Splitting
train.rows<- createDataPartition(y= unbalanced_disease$CASE_CONTROL, p=0.8, list = FALSE)
unbalanced_disease_train<- unbalanced_disease[train.rows,] # 80% data for training
unbalanced_disease_test<-unbalanced_disease[-train.rows,] # 20% data for testing

# Model Traning
############## Elastic Net with Under resampling techniques ####################

x_un_el<-unbalanced_disease_train[,-which(names(unbalanced_disease_train) %in% c("CASE_CONTROL"))]
y_un_el<-unbalanced_disease_train$CASE_CONTROL

down_caret_el<-downSample(x_un_el,y_un_el,yname = "CASE_CONTROL")

elastic_under_el <- train(
  CASE_CONTROL ~., data = down_caret_el, method = "glmnet",
  trControl = trainControl("cv", number = 5),
  tuneLength = 10)

predictions_el<- elastic_under_el %>% predict(unbalanced_disease_test)
predict_el<-predict(elastic_under_el, unbalanced_disease_test,type="prob")[,2]
un_el_downcaret<-metrics(predictions_el,predict_el,unbalanced_disease_test$CASE_CONTROL)

########################## TSMA_EN##############################################
#Prepare for the Aggregation step
estimated_probabilities_under<- matrix(NA, nrow =nrow(unbalanced_disease_test),ncol= 100)

for(j in 1:100){
  # Stage 1 Bootstrap Samples
  train_sample<-unbalanced_disease_train[sample(nrow(unbalanced_disease_train), size = round(0.90 * nrow(unbalanced_disease_train))), ]
  
  # Resampling
  x_un<-train_sample[,-which(names(train_sample) %in% c("CASE_CONTROL"))]
  y_un<-train_sample$CASE_CONTROL
  
  down_caret<-downSample(x_un,y_un,yname = "CASE_CONTROL")
  
  # Model Training
  elastic_under <- train(
    CASE_CONTROL~., data = down_caret, method = "glmnet",
    trControl = trainControl("cv", number = 5),
    tuneLength = 10)
  
  predicted_under<-predict(elastic_under, unbalanced_disease_test,type="prob")[,2]
  estimated_probabilities_under[,j ] <- predicted_under
}

#Aggregation and Prediction
pred.prob <- rowMeans(estimated_probabilities_under, na.rm = TRUE)
pred.class <- as.factor(ifelse(pred.prob > 0.5, 1, 0))
un_el_tsma<-metrics(pred.class,pred.prob,unbalanced_disease_test$CASE_CONTROL)

# Result
compare_result<-data.frame(rbind(un_el_downcaret ,un_el_tsma))
colnames(compare_result) = c("Accuracy","Precision","Recall","Specificity","Balanced Accuracy",
                             "F1_measure ","F2_measure","AUCPR","Cohen's Kappa","AUROC","MCC","G-Mean")
rownames(compare_result) = c("EN with Under", "TSMA-EN")
compare_result


