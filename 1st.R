#author kumar abhinav
#loading train and test dataset
library(readr)
setwd("C:/Users/Lenovo/Downloads/tds/imbalaced data")
test <- read_csv("C:/Users/Lenovo/Downloads/tds/imbalaced data/test.csv")
train <- read_csv("C:/Users/Lenovo/Downloads/tds/imbalaced data/train.csv")
library(data.table)
train <- fread("train.csv",na.strings = c(""," ","?","NA",NA))
test <- fread("test.csv",na.strings = c(""," ","?","NA",NA))
train[1:5]
unique(train$income_level)
unique(test$income_level)

#encode target variables
train$income_level<-ifelse(train$income_level=='-50000',0,1)
test$income_level<-ifelse(test$income_level=='-50000',0,1)
prop.table(table(train$income_level))*100
round(prop.table(table(train$income_level))*100)
#0  1 
#94  6 
str(train)
#set column classes into factors and num
factcols <- c(2:5,7,8:16,20:29,31:38,40,41)
numcols<-setdiff(1:40,factcols)
as.list.data.frame(factcols)
as.list.data.frame(numcols)

train[factcols]<-lapply(train[factcols],as.factor)
train[numcols]<-lapply(train[numcols],as.numeric)
test[factcols]<-lapply(test[factcols],as.factor)
test[numcols]<-lapply(test[numcols],as.numeric)
#==========================================================================================\
#                     data exploration                                                      \
#============================================================================================\
#separate categorical variables & numerical variables
cat_train<-subset(train,select=c(factcols))
cat_test<-subset(test,select=c(factcols))
num_train<-subset(train,select=c(numcols))
num_test<-subset(test,select=c(numcols))
#saving some memory
rm(test)
rm(train)
library(ggplot2)
install.packages("plotly")
library(plotly)
summary(num_train)
plot(num_train$age)
library(pastecs)
#get datailed summary of continuos variables
stat.desc(num_train)
#all numerical variables must be analyzed through histogram
tr <- function(a){
  ggplot(data = num_train, aes(x= a, y=..density..)) +
  geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100)+
  geom_density()
}

tr(num_train$age)
#variable capital_losses
tr(num_train$capital_losses)
#variable weeks worked in year
tr(num_train$weeks_worked_in_year)
#analysing count % of cat variables
table(cat_train$occupation_code)
as.matrix(prop.table(table(cat_train$occupation_code)))
#anlyzing with scatter plot bet. numerical and target
num_train$income_level<-cat_train$income_level
#create a scatter plot ie multi variate analysis of two num var.
ggplot(data=num_train,aes(x = age, y=wage_per_hour))+geom_point(aes(colour=income_level))+
  scale_y_continuous("wage per hour", breaks = seq(0,10000,1000))
#analyzing cat and target using bar plot
all_bar<- function(i){
  ggplot(cat_train,aes(x=i,fill=income_level))+geom_bar(position = "dodge", 
  color="black")+scale_fill_brewer(palette = "Pastel1")+
    theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))
}
  
all_bar(cat_train$class_of_worker)
all_bar(cat_train$education) 
#2 way tables
prop.table(table(cat_train$marital_status,cat_train$income_level),1)  
prop.table(table(cat_train$class_of_worker,cat_train$income_level),1)
#another way to look for 2 cat var. is using gmodels
library(gmodels)
CrossTable(cat_train$class_of_worker,cat_train$income_level)  
CrossTable(cat_train$marital_status,cat_train$income_level)
CrossTable(cat_train$education,cat_train$income_level) 
  
#==========================================================================================\
#                     data cleaning                                                         \
#============================================================================================\
#check missing values in numerical data
table(is.na(num_train))
table(is.na(num_test))
colSums(is.na(num_train))
colSums(is.na(num_train))
library(caret)
#set threshold as 0.7
ax <-findCorrelation(x = cor(num_train), cutoff = 0.7)
#removing the cols with higher correlation than threshold
num_train<-subset(num_train,select =-c(ax))
num_test<-subset(num_test,select=-c(ax))
#check missing values per columns
mvtr <- sapply(cat_train, function(x){sum(is.na(x))/length(x)})*100
mvte <- sapply(cat_test, function(x){sum(is.na(x)/length(x))}*100)
mvtr
mvte
plot(mvte)
plot(mvtr)
#select columns with missing value less than 5%
cat_train <- subset(cat_train, select = mvtr < 5 )
cat_test <- subset(cat_test, select = mvte < 5)

#set NA as Unavailable - train data
#convert to characters
cat_train <- cat_train[,names(cat_train) := lapply(.SD, as.character),.SDcols = names(cat_train)]
for (i in seq_along(cat_train)) set(cat_train, i=which(is.na(cat_train[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_train <- cat_train[, names(cat_train) := lapply(.SD,factor), .SDcols = names(cat_train)]

#set NA as Unavailable - test data
cat_test <- cat_test[, (names(cat_test)) := lapply(.SD, as.character), .SDcols = names(cat_test)]
for (i in seq_along(cat_test)) set(cat_test, i=which(is.na(cat_test[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_test <- cat_test[, (names(cat_test)) := lapply(.SD, factor), .SDcols = names(cat_test)]

#==========================================================================================\
#                     data manipulation                                                     \
#============================================================================================\

#combine factor levels with less than 5% values
#cat_train
for(i in names(cat_train)){
  p <- 5/100
  ld <- names(which(prop.table(table(cat_train[[i]])) < p))
  levels(cat_train[[i]])[levels(cat_train[[i]]) %in% ld] <- "Other"
}
#cat_test
for(i in names(cat_test)){
  p <- 5/100
  ld <- names(which(prop.table(table(cat_test[[i]])) < p))
  levels(cat_test[[i]])[levels(cat_test[[i]]) %in% ld] <- "Other"
}

#check columns with unequal levels 
library(mlr)
summarizeColumns(cat_train)[,"nlevs"]
summarizeColumns(cat_test)[,"nlevs"]
#counts of unique values in these variables 
num_train[,.N,age][order(age)]
num_train[,.N,wage_per_hour][order(-N)]

#bin age variable 0-30 31-60 61 - 90
num_train[,age:= cut(x = age,breaks = c(0,30,60,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_train[,age := factor(age)]

num_test[,age:= cut(x = age,breaks = c(0,30,60,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_test[,age := factor(age)]

num_train[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_train[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_train[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_train[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

num_test[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_test[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_test[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_test[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]
num_train[,income_level := NULL]
#combine data and make test & train files
d_train <- cbind(num_train,cat_train)
d_test <- cbind(num_test,cat_test)

#remove unwanted files
rm(num_train,num_test,cat_train,cat_test) 

#==========================================================================================\
#                     predictive modelling                                                  \
#============================================================================================\

#create task
train.task <- makeClassifTask(data = d_train,target = "income_level")
test.task <- makeClassifTask(data=d_test,target = "income_level")
train.task
test.task

#remove zero variance features
train.task <- removeConstantFeatures(train.task)
test.task <- removeConstantFeatures(test.task)

#normalize the variables
train.task<- normalizeFeatures(train.task,method = "standardize")
test.task<- normalizeFeatures(test.task,method = "standardize")
#get variable importance chart
install.packages("FSelector",dependencies = TRUE)
library(FSelector)

var_imp <- generateFilterValuesData(train.task, method = c("information.gain"))
plotFilterValues(var_imp,feat.type.cols = TRUE)

#methods to deal with imbalanced as balanced data
#undersampling 
train.under <- undersample(train.task,rate = 0.1) #keep only 10% of majority class
table(getTaskTargets(train.under))

#0     1 
#18711 12416 

#oversampling
train.over <- oversample(train.task,rate=15) #make minority class 15 times
table(getTaskTargets(train.over))

#0      1 
#187107 186240 
#more good results in oversampling
#SMOTE synthetic minority oversampling technique
system.time(
  train.smote <- smote(train.task,rate =4,nn = 3) 
) 
table(getTaskTargets(train.smote))
#this shows how much memory and disk it can take

#lets see which algorithms are available
listLearners("classif","twoclass")[c("class","package")]
#naive Bayes
naive_learner <- makeLearner("classif.naiveBayes",predict.type = "response")
naive_learner$par.vals <- list(laplace = 1)

#10fold CV - stratified
folds <- makeResampleDesc("CV",iters=10,stratify = TRUE)

#cross validation function
fun_cv <- function(a){
  crv_val <- resample(naive_learner,a,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))
  crv_val$aggr
}

#comparing different tasks
fun_cv (train.task)
#acc.test.mean   tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
#0.7319256       0.7210847     0.8952935     0.1047065   130.0000000  5218.7000000 
fun_cv(train.under) 
#acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
#0.76354911    0.66436834    0.91301383    0.08698617  108.00000000  628.00000000 
fun_cv(train.over)
#acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
#0.7848838e    0.6553042       0.9150666       0.8493342       0.1581800    0.6449500
fun_cv(train.smote)
#acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
#0.7320359     0.7212023     0.8952945     0.1047055   130.0000000  5216.5000000 

#so highest +ve and -ve in smote
#train and predict
nB_model <- caret::train(naive_learner, train.smote)
nB_predict <- predict(nB_model,test.task)
