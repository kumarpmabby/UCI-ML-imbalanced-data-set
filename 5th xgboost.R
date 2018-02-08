#author kumar abhinav
#xgboost
library(readr)
train <- read_csv("C:/Users/Lenovo/Downloads/tds/imbalaced data/train.csv")
test <- read_csv("C:/Users/Lenovo/Downloads/tds/imbalaced data/test.csv")
train[sapply(train,is.character)]<-lapply(train[sapply(train,is.character)],as.factor)
test[sapply(train,is.character)]<-lapply(test[sapply(train,is.character)],as.factor)
train$income_level<-ifelse(train$income_level=='-50000',0,1)
test$income_level<-ifelse(test$income_level=='-50000',0,1)


library(xgboost)
param <- list(
  "objective"  = "binary:logistic",
  "eval_metric" = "auc",
  "eta" = 0.01 
  ,"subsample" = 1
  , "colsample_bytree" = 1
  , "min_child_weight" = 1
  , "max_depth" = 9
) 
xgtrain= as.matrix(train)
mode(xgtrain) = "numeric"
#xgtrain <- xgb.DMatrix(as.matrix(train), label = y)

