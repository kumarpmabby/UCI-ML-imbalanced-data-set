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
#using rose package to make balanced from imbalanced
library(ROSE)
prop.table(table(d_train$income_level))
d_train$income_level<-ifelse(d_train$income_level=='-50000',0,1)
d_test$income_level<-ifelse(d_test$income_level=='-50000',0,1)
library(rpart)
treeimb <- rpart(d_train$income_level ~ ., data = d_train)
pred.treeimb <- predict(treeimb, newdata =d_test)
plot(pred.treeimb)

accuracy.meas(d_test$income_level, pred.treeimb)

#Examples are labelled as positive when predicted is greater than 0.5 

#precision: 0.606
#recall: 0.442
#F: 0.256
#which is not that much great
roc.curve(d_test$income_level,pred.treeimb,plotit = T)
#Area under the curve (AUC): 0.870
#oversample
data_balanced_over<-ovun.sample(income_level~.,data=d_train,method = "over",N=374282)$data
table(data_balanced_over$income_level)
#0      1 
#187141 187141 
#undersample
data_balanced_under<-ovun.sample(income_level~.,data=d_train,method = "under",N=24764)$data
table(data_balanced_under$income_level)
#0     1 
#12382 12382 
data.rose<-ROSE(income_level~.,data=d_train,seed=1)$data
table(data.rose$income_level)
#0     1 
#99614 99909 
#so rose has perfectly balanced
#lets predict the model on those
tree.rose <- rpart(income_level ~ ., data = data.rose)
tree.over <- rpart(income_level ~ ., data = data_balanced_over)
tree.under <- rpart(income_level ~ ., data = data_balanced_under)

#make predictions
pred.tree.rose <- predict(tree.rose, newdata = d_test)
pred.tree.over <- predict(tree.over, newdata = d_test)
pred.tree.under <- predict(tree.under, newdata = d_test)

#find AUC for all
roc.curve(d_test$income_level, pred.tree.rose,plotit = T ,col="green")
#Area under the curve (AUC): 0.848

#AUC Oversampling
roc.curve(d_test$income_level, pred.tree.over,plotit = T ,col="yellow")
#Area under the curve (AUC): 0.891

#AUC Undersampling
roc.curve(d_test$income_level, pred.tree.under,plotit = T ,col="orange")
#Area under the curve (AUC): 0.892

#xgboost
require(xgboost)
set.seed(1)
param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss", "colsample_bytree" = 1,
              "min_child_weight" = 1,
              "eta" = 0.01, "max.depth" = 9,"subsample"=1)

#making as DMatrix ie. large matrix
dtrain= as.matrix(d_train)
mode(dtrain) = "numeric"

dtest= as.matrix(d_test)
mode(dtest) = "numeric"

dtrain <- xgb.DMatrix(data = d_train$data, label=d_train$income_level)
dtest <- xgb.DMatrix(data = d_test$data, label=d_test$income_level)


bst.cv = xgb.cv(param=param,data =dtrain,prediction=TRUE, verbose=T,showsd = T,
                stratified = T, print.every.n = 10,
                label = d_train$income_level,nfold = 10, nrounds = 50)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#here we see that observation that after 44th round there is no change in log loss
plot(log(bst.cv$test.logloss.mean),type = "l")
watchlist <- list(train=dtrain, test=dtest)
#real model
bst <- xgb.train(data =dtrain, label = d_train$income_level, max.depth =9, eta =0.01, nrounds = 150,
               nthread = 2)

preds=predict(bst,dtest)
trees = xgb.model.dt.tree(colnames(dtrain),model = bst)
# Get the feature real names
names <- dimnames(dtrain)
importance_matrix <- xgb.importance(names, model = bst)
gp=xgb.plot.importance(importance_matrix[1:20])
print(gp) 

xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 4)
