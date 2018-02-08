#author kumar abhinav
#loading train and test dataset
library(readr)
setwd("C:/Users/Lenovo/Downloads/tds/imbalaced data")
test <- read_csv("C:/Users/Lenovo/Downloads/tds/imbalaced data/test.csv")
train <- read_csv("C:/Users/Lenovo/Downloads/tds/imbalaced data/train.csv")
library ( caret )
library ( data.table )
library ( xgboost )
train <- fread("train.csv",na.strings = c(""," ","?","NA",NA))
test <- fread("test.csv",na.strings = c(""," ","?","NA",NA))

factcols <- c(2:5,7,8:16,20:29,31:38,40,41)
numcols<-setdiff(1:40,factcols)
train[factcols]<-lapply(train[factcols],as.factor)
train[numcols]<-lapply(train[numcols],as.numeric)
test[factcols]<-lapply(test[factcols],as.factor)
test[numcols]<-lapply(test[numcols],as.numeric)

cat_train<-subset(train,select=c(factcols))
cat_test<-subset(test,select=c(factcols))
num_train<-subset(train,select=c(numcols))
num_test<-subset(test,select=c(numcols))

rm(test)
rm(train)
ax  <- findCorrelation (cor ( num_train ), cutoff  =  0.7)
num_train<-subset(num_train,select =-c(ax))
num_test<-subset(num_test,select=-c(ax))
#check missing values per columns
mvtr <- sapply(cat_train, function(x){sum(is.na(x))/length(x)})*100
mvte <- sapply(cat_test, function(x){sum(is.na(x)/length(x))}*100)
mvtr
mvte
cat_train <- subset(cat_train, select = mvtr < 5 )
cat_test <- subset(cat_test, select = mvte < 5)

cat_train <- cat_train[,names(cat_train) := lapply(.SD, as.character),.SDcols = names(cat_train)]
for (i in seq_along(cat_train)) set(cat_train, i=which(is.na(cat_train[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_train <- cat_train[, names(cat_train) := lapply(.SD,factor), .SDcols = names(cat_train)]

#set NA as Unavailable - test data
cat_test <- cat_test[, (names(cat_test)) := lapply(.SD, as.character), .SDcols = names(cat_test)]
for (i in seq_along(cat_test)) set(cat_test, i=which(is.na(cat_test[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_test <- cat_test[, (names(cat_test)) := lapply(.SD, factor), .SDcols = names(cat_test)]

d_train <- cbind(num_train,cat_train)
d_test <- cbind(num_test,cat_test)
rm(num_train,num_test,cat_train,cat_test) 


#xgboost
#using one hot encoding
tr_labels <- d_train$income_level
ts_labels <- d_test$income_level
new_tr <- model.matrix(~.+0,data = d_train[,-c("income_level"),with=F])
new_ts <- model.matrix(~.+0,data = d_test[,-c("income_level"),with=F])
#convert factor to numeric
tr_labels <- as.numeric(tr_labels)-1
ts_labels <- as.numeric(ts_labels)-1

dtrain <- xgb.DMatrix(data = new_tr,label = tr_labels) 
dtest <- xgb.DMatrix(data = new_ts,label= ts_labels)

params <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, 
               min_child_weight=1, subsample=1, 
               colsample_bytree=1)
xgbcv <- xgb.cv( params = params, 
                 data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, 
                 stratified = T, print.every.n = 10,
                 early.stop.round = 2, maximize = F)
xgb1 <- xgb.train (params = params, 
                   data = dtrain, nrounds = 10, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print.every.n = 10, 
                   early.stop.round = 10, 
                   maximize = F , eval_metric = "error")
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)