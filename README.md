---
title: "Weight Lifting Exercises - \"How Well\" Recognition"
author: "Jane Chen"
date: "April 12, 2017"
output: html_document
---

```{r dataLoading}
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

install.packages("caret", repo = "https://cran.r-project.org"); library(caret)
install.packages("randomForest", repo = "https://cran.r-project.org"); library(randomForest)
```

## Introduction
This is a Write-Up report for Coursera Practical Machine Learning (from Data Science specialization) Peer-Reviewed Assessment. The aim of this project is to analysis the training dataset and create a prediction model in how well the user has performed the Unilateral Dumbbell Biceps Curl by using the sensors. 

Each class (`classe` in the datasets) indicates different classes of "how-well" the participants have performed the bicep curls. 
- Class A: exactly according to the specification
- Class B: throwing the elbows to the front
- Class C: lifting the dumbbell only halfway
- Class D: lowering the dumbbell only halfway
- Class E: throwing the hips to the front

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4eRMLOkyO

## Method
```{r dataProcessing, results = FALSE, message = FALSE,warning = FALSE}
# Point A: remove the index, username, timestamps and windows
training <- training[, -c(1:7)]
# Point B: remove columns where most observations don't have the result for (NA)
factorCol <- logical(ncol(training))
for (i in 1:(ncol(training)-1)) {
        factorCol[i] <- (class(training[,i ]) == "factor")
}
numTrain <- training
for (i in (1:ncol(training))[factorCol]) {
        numTrain[, i] <- as.numeric(as.character(training[, i]))
}
wanted <- logical(ncol(numTrain))
for (i in 1:ncol(numTrain)) {
        wanted[i] <- (sum(is.na(numTrain[, i])) == 0)
}
newTrain <- numTrain[, wanted]
str(newTrain)
# Point C: create train, test and validation sets
set.seed(12345)
inTrain <- createDataPartition(y = newTrain$classe, p = 0.6, list = FALSE)
subTrain <- newTrain[inTrain, ]
subTest <- newTrain[-inTrain, ]
```
The first few columns, including index, uswername, time staps and window, were removed as they were unrelated to the prediction(see point A from the R code above). Variables with mostly `NA`s were also removed as the second step (see point B). The remaining training dataset was then separated into `subTrain` and `subTest` to train the model and predict out of sample error rate (see point C). 

### Model Selection and Cross Validation
```{r models, echo = FALSE, message = FALSE, warning = FALSE}
# Point D: model with "Random Forest"(rf) with 5-fold cross validation
set.seed(9876)
fold <- createFolds(y = subTrain$classe, k = 5, list = FALSE)
fold1_train <- subTrain[fold != 1, ]
fold1_test <- subTrain[fold == 1, ]
fold2_train <- subTrain[fold != 2, ]
fold2_test <- subTrain[fold == 2, ]
fold3_train <- subTrain[fold != 3, ]
fold3_test <- subTrain[fold == 3, ]
fold4_train <- subTrain[fold != 4, ]
fold4_test <- subTrain[fold == 4, ]
fold5_train <- subTrain[fold != 5, ]
fold5_test <- subTrain[fold == 5, ]
mdl_rf1 <- randomForest(classe ~ ., data = fold1_train, method = "class")
pred_rf1 <- predict(mdl_rf1, fold1_test)
mdl_rf2 <- randomForest(classe ~ ., data = fold2_train, method = "class")
pred_rf2 <- predict(mdl_rf2, fold2_test)
mdl_rf3 <- randomForest(classe ~ ., data = fold3_train, method = "class")
pred_rf3 <- predict(mdl_rf3, fold3_test)
mdl_rf4 <- randomForest(classe ~ ., data = fold4_train, method = "class")
pred_rf4 <- predict(mdl_rf4, fold4_test)
mdl_rf5 <- randomForest(classe ~ ., data = fold5_train, method = "class")
pred_rf5 <- predict(mdl_rf5, fold5_test)
# Point E: model with "Linear Discriminant Analysis"(lda) with 5-fold cross validation
mdl_lda1 <- train(classe ~ ., data = fold1_train, method = "lda")
pred_lda1 <- predict(mdl_lda1, fold1_test)
mdl_lda2 <- train(classe ~ ., data = fold2_train, method = "lda")
pred_lda2 <- predict(mdl_lda2, fold2_test)
mdl_lda3 <- train(classe ~ ., data = fold3_train, method = "lda")
pred_lda3 <- predict(mdl_lda3, fold3_test)
mdl_lda4 <- train(classe ~ ., data = fold4_train, method = "lda")
pred_lda4 <- predict(mdl_lda4, fold4_test)
mdl_lda5 <- train(classe ~ ., data = fold5_train, method = "lda")
pred_lda5 <- predict(mdl_lda5, fold5_test)
```
In this analysis, two prediction modelling were used, random forest (RF) and linear discriminant analysis (LDA). RF was chosen to represent the machine learning algorithm for classification, and LDA was chosen to represent the machine learning algorithm for linear combination and pattern recognition. With 5-fold cross validation, the out of sample error rate for RF are much smaller than that of LDA (accuracy are approximately 0.99 and 0.7  respectively, see code and result in appendix). Hence, the model selected here is RF. 

```{r echo = FALSE, warning = FALSE, message = FALSE}
mdl_rf <- randomForest(classe ~ ., data = subTrain, method = "class")
pred_rf <- predict(mdl_rf, subTest)
acc <- confusionMatrix(pred_rf, subTest$classe)$overall[1]
outSmpEr <- 1 - acc
```

The out of sample error is approximately `r outSmpEr` for the random forest model. 

## Result
```{r prediction}
finalMdl <- randomForest(classe ~ ., data = newTrain, method = "class")
prediction <- predict(finalMdl, testing)
```
The predictions for the testing dataset are `r prediction`. 

## Appendix
### 5-Fold Cross Validation for Random Forest Prediction
```{r rfcv}
mdl_rf1 <- randomForest(classe ~ ., data = fold1_train, method = "class")
pred_rf1 <- predict(mdl_rf1, fold1_test)
confusionMatrix(pred_rf1, fold1_test$classe)$overall[1]
mdl_rf2 <- randomForest(classe ~ ., data = fold2_train, method = "class")
pred_rf2 <- predict(mdl_rf2, fold2_test)
confusionMatrix(pred_rf2, fold2_test$classe)$overall[1]
mdl_rf3 <- randomForest(classe ~ ., data = fold3_train, method = "class")
pred_rf3 <- predict(mdl_rf3, fold3_test)
confusionMatrix(pred_rf3, fold3_test$classe)$overall[1]
mdl_rf4 <- randomForest(classe ~ ., data = fold4_train, method = "class")
pred_rf4 <- predict(mdl_rf4, fold4_test)
confusionMatrix(pred_rf4, fold4_test$classe)$overall[1]
mdl_rf5 <- randomForest(classe ~ ., data = fold5_train, method = "class")
pred_rf5 <- predict(mdl_rf5, fold5_test)
confusionMatrix(pred_rf5, fold5_test$classe)$overall[1]
```
### 5-Fold Cross Validation for Linear Discriminant Analysis Prediction
```{r ldacv}
mdl_lda1 <- train(classe ~ ., data = fold1_train, method = "lda")
pred_lda1 <- predict(mdl_lda1, fold1_test)
confusionMatrix(pred_lda1, fold1_test$classe)$overall[1]
mdl_lda2 <- train(classe ~ ., data = fold2_train, method = "lda")
pred_lda2 <- predict(mdl_lda2, fold2_test)
confusionMatrix(pred_lda2, fold2_test$classe)$overall[1]
mdl_lda3 <- train(classe ~ ., data = fold3_train, method = "lda")
pred_lda3 <- predict(mdl_lda3, fold3_test)
confusionMatrix(pred_lda3, fold3_test$classe)$overall[1]
mdl_lda4 <- train(classe ~ ., data = fold4_train, method = "lda")
pred_lda4 <- predict(mdl_lda4, fold4_test)
confusionMatrix(pred_lda4, fold4_test$classe)$overall[1]
mdl_lda5 <- train(classe ~ ., data = fold5_train, method = "lda")
pred_lda5 <- predict(mdl_lda5, fold5_test)
confusionMatrix(pred_lda5, fold5_test$classe)$overall[1]
```
