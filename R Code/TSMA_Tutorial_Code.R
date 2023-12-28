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
  MCC<-mcc(pred.class,ref); met[11] = MCC
  G_mean<-sqrt(precision *specificity );met[12]=G_mean
  
  return(met)
}

# Loading the Data
```{r}
camrsa <- readRDS("~/Documents/camrsa/Data/camrsa.rds")
camrsa[1:5,1:6]
table(camrsa$CASE_CONTROL_CAMRSA)

# Data Splitting
train.rows<- createDataPartition(y= camrsa$CASE_CONTROL_CAMRSA, p=0.8, list = FALSE)
camrsa_train<- camrsa[train.rows,][,-1] # 80% data for training
camrsa_test<-camrsa[-train.rows,][,-1] # 20% data for testing

# Model Traning
############## Elastic Net with Under resampling techniques ####################
```{r eval=FALSE}
x_ca_el<-camrsa_train[,-which(names(camrsa_train) %in% c("CASE_CONTROL_CAMRSA"))]
y_ca_el<-camrsa_train$CASE_CONTROL_CAMRSA

down_caret_el<-downSample(x_ca_el,y_ca_el,yname = "CASE_CONTROL_CAMRSA")

elastic_under_el <- train(
  CASE_CONTROL_CAMRSA ~., data = down_caret_el, method = "glmnet",
  trControl = trainControl("cv", number = 5),
  tuneLength = 10)

predictions_el<- elastic_under_el %>% predict(camrsa_test)
predict_el<-predict(elastic_under_el, camrsa_test,type="prob")[,2]
ca_el_downcaret<-metrics(predictions_el,predict_el,camrsa_test$CASE_CONTROL_CAMRSA)

########################## TSMA_EN##############################################
#Prepare for the Aggregation step
estimated_probabilities_under<- matrix(NA, nrow =nrow(camrsa_test),ncol= 100)

for(j in 1:100){
  # Stage 1 Bootstrap Samples
  train_sample<-camrsa_train[sample(nrow(camrsa_train), size = round(0.90 * nrow(camrsa_train))), ]
  
  # Resampling
  x_ca<-train_sample[,-which(names(train_sample) %in% c("CASE_CONTROL_CAMRSA"))]
  y_ca<-train_sample$CASE_CONTROL_CAMRSA
  
  down_caret<-downSample(x_ca,y_ca,yname = "CASE_CONTROL_CAMRSA")
  
  # Model Training
  elastic_under <- train(
    CASE_CONTROL_CAMRSA ~., data = down_caret, method = "glmnet",
    trControl = trainControl("cv", number = 5),
    tuneLength = 10)
  
  predicted_under<-predict(elastic_under, camrsa_test,type="prob")[,2]
  estimated_probabilities_under[,j ] <- predicted_under
}

#Aggregation and Prediction
pred.prob <- rowMeans(estimated_probabilities_under, na.rm = TRUE)
pred.class <- as.factor(ifelse(pred.prob > 0.5, 1, 0))
ca_el_tsma<-metrics(pred.class,pred.prob,camrsa_test$CASE_CONTROL_CAMRSA)

# Result
compare_result<-data.frame(rbind(ca_el_downcaret ,ca_el_tsma))
colnames(compare_result) = c("Accuracy","Precision","Recall","Specificity","Balanced Accuracy",
                             "F1_measure ","F2_measure","AUCPR","Cohen's Kappa","AUROC","MCC","G-Mean")
rownames(compare_result) = c("EN with Under", "TSMA-EN")
compare_result



