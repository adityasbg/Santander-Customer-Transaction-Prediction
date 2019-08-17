  library(gridExtra)
  library(grid)
  library(ggplot2)
  library(lattice)
  library(usdm)
  library(pROC)
  library(caret)
  library(rpart)
  library(DataCombine)
  library(ROSE)
  library(e1071)
  library(xgboost)
  
  setwd("/home/aditya/code_pen/edwiser/project 2/r/sandler")
  #Reading test and train data frame
  train =read.csv('train.csv')
  test =read.csv('test.csv')
  #checking dimension of train dataset
  dim(train)
  #checking dimension of test dataset
  dim(test)
  
  #basic Descriptive stats 
  summary(train)
  summary(test)
  ################### obsevations ######################################################################
  # most of the distribution mean and median are almost same
  
  #storing ID_code  of test train data 
  train_ID_code_orignal = train$ID_code
  test_Id_code_orignal  = test$ID_code
  
  #removing Idcode from orginal dataset 
  train$ID_code=NULL
  test$ID_code=NULL
  
  #check dimension of dataset after removing column
  print(dim(train))
  print(dim(test))
  
  #count of target variable 
  table(train$target)
  
  ###############################################################################################################
  #################################################################################################################
  
  
  #Missing value analysis
  # this function takes dataframe as input and calulate precentage of missing values in each 
  # columns and returns that dataframe 
  
  findMissingValue =function(df){
    missing_val =data.frame(apply(df,2,function(x){sum(is.na(x))}))
    missing_val$Columns = row.names(missing_val)
    names(missing_val)[1] =  "Missing_percentage"
    missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
    missing_val = missing_val[order(-missing_val$Missing_percentage),]
    row.names(missing_val) = NULL
    missing_val = missing_val[,c(2,1)]
    return (missing_val)
  }
  
  #check missing value in train dataset
  head(findMissingValue(train))
  #check missing value in test dataset
  head(findMissingValue(test))
  
  ############ No missing value in test and train data #########################################
  
  
  
  # creating target and independent variable from train dataset
  independent_var= (colnames(train)!='target')
  X=train[,independent_var]
  Y=train$target
  
  
  #Multicolinearity Analysis
  #checking is variable are correlated
  cor=vifcor(X)
  print(cor)
  
  
  ################### No varible are correlated #####################################################
  
  
  # Distribution plot
  #This function plots distribution plot from given data set
  plot_distribution =function(X)
  {
    variblename =colnames(X)
    temp=1
    for(i in seq(10,dim(X)[2],10))
    {
      plot_helper(temp,i ,variblename)
      temp=i+1
    }
  }
  
  # helper function takes start and stop index to print subset distribution plot
  plot_helper =function(start ,stop, variblename)
  { 
    par(mar=c(2,2,2,2))
    par(mfrow=c(4,3))
    for (i in variblename[start:stop])
    {
      plot(density(X[[i]]) ,main=i )
    }
  }
  
  
  # plot density plot for trainig data 
  plot_distribution(X)
  
  
  ###################Observation distribution  train dataset  ################################################################
  #  Allmost all Distributions of variables are normal
  
  #plot density plot for testing data
  plot_distribution(test)
  
  ###########Observation  distribution Test data #################################################################
  # Allmost all Distributions of variables are normal
  # Test data is very similar to train data in terms of distribution
  
  
  ###################################################################################################################
  ###############################################  Outliers #######################################################
  
  #This function plots boxplot plot from given data set
  #X =dataframe
  plot_boxplot =function(X)
  {
    variblename =colnames(X)
    temp=1
    for(i in seq(10,dim(X)[2],10))
    {
      plot_helper(temp,i ,variblename)
      temp=i+1
    }
  }
  
  # helper function takes start and stop index to print subset distribution plot
  plot_helper =function(start ,stop, variblename)
  { 
    par(mar=c(2,2,2,2))
    par(mfrow=c(4,3))
    for (i in variblename[start:stop])
    {
      boxplot(X[[i]] ,main=i)
    }
  }
  
  
  #boxplot for training data
  plot_boxplot(X)
  
  #box plot for testing data 
  plot_boxplot(test)
  
  
  # This function takes dataframe as input and fill outliers with null and return modified dataframe
  # df = dataframe input 
  fill_outlier_with_na=function(df)
  {
    cnames=colnames(df)
    for(i in cnames)
    {
      
      val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
      df[,i][df[,i] %in% val] = NA
    }
    return (df)
  }
  
  
  #remove outlier from train data and fill na's
  X=fill_outlier_with_na(X)
  print(paste0("Total na's in training data ::" ,sum(is.na(X))))
  
  #remove outlier from test data fill na's
  test=fill_outlier_with_na(test)
  print(paste0("Total na's in testing data ::" ,sum(is.na(test))))
  
  
  # This function takes dataframe as input and fill outliers with null and return modified dataframe
  #df = dataframe input 
  fill_outlier_with_mean=function(df)
  {
    cnames=colnames(df)
    for(i in cnames)
    {
      
      df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
    }
    return (df)
  }
  
  
  # fill na's with mean
  X=fill_outlier_with_mean(X)
  print(paste0("Total na's in training data ::" ,sum(is.na(X))))
  
  # fill na's with mean 
  test=fill_outlier_with_mean(test)
  print(paste0("Total na's in testing data ::" ,sum(is.na(test))))
  #mean imputation done
  
  
  
  ##################################### Standardisation ##########################################################
  
  # This function takes data frame as input and standardize dataframe
  # df =data frame 
  # formula=(x=mean(x))/sd(x)
  standardizing=function(df)
  {
    cnames =colnames(df)
    for( i in   cnames ){
      df[,i]=(df[,i] -mean(df[,i] ,na.rm=T))/sd(df[,i])
    }
    return(df)
    
  }
  
  #standardize train data 
  X=standardizing(X)
  
  #standardise test data
  test =standardizing(test)
  
  # combine independent and dependent variables
  std_train =cbind(X,Y)
  
  # spilt in test train set 
  # create stratified sampling
  # 70% data in train in training set 
  set.seed(123)
  train.index =createDataPartition(std_train$Y , p=.70 ,list=FALSE)
  train = std_train[train.index,]
  test  = std_train[-train.index,]
  
  #random over sampling keeping 0's and 1's (50 :50 ) sample 
  over= ovun.sample(Y~. ,data =train  , method='over' )$data
  
  # print dim of data afer partition 
  print("dim train data")
  dim(train)
  print("dim test data ")
  dim(test)

  getmodel_accuracy=function(conf_matrix)
  {
    model_parm =list()
    tn =conf_matrix[1,1]
    tp =conf_matrix[2,2]
    fp =conf_matrix[1,2]
    fn =conf_matrix[2,1]
    p =(tp)/(tp+fp)
    r =(fp)/(fp+tn)
    f1=2*((p*r)/(p+r))
    print(paste("accuracy",round((tp+tn)/(tp+tn+fp+fn),2)))
    print(paste("precision",round(p ,2)))
    print(paste("recall",round(r,2)))
    print(paste("fpr",round((fp)/(fp+tn),2)))
    print(paste("fnr",round((fn)/(fn+tp),2)))
    print(paste("f1",round(f1,2)))
    
  }
  ###############################################################################################################
  ############################ Model training ####################################################################
  
  # this function takes confusion matrix as input and print various classification  metrics
 

  ##########################################################################################################
  ################################# LOGISTIC REGRESSION #################################################### 
  
  
  
  #fitting logistic model BASE
  over_logit =glm(formula = Y~. ,data =train ,family='binomial')
  # model summary  
  summary(over_logit)
  #get model predicted  probality 
  y_prob =predict(over_logit , test[-201] ,type = 'response' )
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  # get auc 
  roc=roc(test[,201], y_prob)
  print(roc )
  # plot roc _auc plot 
  plot(roc ,main ="Logistic Regression base Roc ")
 ################## model 1 ########################################### 
  ################## test data prediction   ##########################################
  #[1] "accuracy 0.92"
  #[2] "precision 0.68"
  #[3] "recall 0.01"
  #[4] "fpr 0.01"
  #[5] "fnr 0.73"
  #[6] "f1 0.03"
  #Area under the curve: 0.8585
  
  
  # accuracy of model is very good 92%
  # model has very low recall 1% only
  # f1 score is just 3%
  # very low very poor model
  
  ############################## OVER SAMPLING #################################################################
  

  
  #fitting logistic model
  over_logit =glm(formula = Y~. ,data =over ,family='binomial')
  # model summary  
  summary(over_logit)
  #PREDICTING PROBALITY
  y_prob =predict(over_logit , test[-201] ,type = 'response' )
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  #get Auc 
  roc=roc(test[,201], y_prob )
  print(roc)
  #plot roc curve 
  plot(roc ,main="Logistic Regression roc-auc oversampled")
  #$$$$$
  ####### test data prediction  ##################
  #[1] "accuracy 0.78"
  #[2] "precision 0.28"
  #[3] "recall 0.22"
  #[4] "fpr 0.22"
  #[5] "fnr 0.22"
  #[6] "f1 0.25"
  #Area under the curve: 0.8582
  ################### observation model 2  ###################################
  # accuracy of model is ok 78%  but has decreased from 92%
  #fpr and fnr rate is 22% which is ok
  #f1 score is 25% which is very low  but base model had only 3%
  #auc is 85% which is good 
  #overall better than base logistic function 
  
  
  
  
  
  ################################## Decision Tree #########################################################
  #########################################################################################################
  
  
  ################################## Model 1 ###############################################################
  ################### using over sampled data  ############################################################
  

  #train model
  rm_model = rpart( Y~. ,data =over )
  # model summary  
  summary(rm_model)
  #PREDICTING PROBALITY
  y_prob =predict(rm_model , test[-201] )
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  #get Auc score 
  roc=roc(test[,201], y_prob )
  print(roc)
  # plot roc_auc curve 
  plot(roc ,main="Roc _ auc Decision tree (Over sampled)")
  ####################### test data  model 1  ###########################################
  # [1] "accuracy 0.63"
  # [1] "precision 0.14"
  # [1] "recall 0.35"
  # [1] "fpr 0.35"
  # [1] "fnr 0.47"
  # [1] "f1 0.2"
  #Area under the curve: 0.5936
  
  
  ################ Observation  model 1 ##########################################################################
  # model perform poorer  than Logistic regression model 
  # f1 score is only 20%
  # fnr rate is 47 % which is high 
  # precision recall is very poor 
  # Roc _ auc  around 59%  over all model is very poor 
  
  
  
  ########################## Tuning Parameter (model 2)  ##############################################
  # tuning param 
  #cp complexity parameter. Any split that does not decrease the overall lack of fit by a factor of cp is not attempted
  
  # converting target varible o's and 1's  to 'no' 'yes '
  over$Y =ifelse(over$Y ==0,'No','yes')
  # train control 
  ctrl=trainControl(classProbs = TRUE , summaryFunction = twoClassSummary)
  # get best model that has highest roc value
  rm_model <- train(
    Y ~ .,
    data = over,
    method =  'rpart',
    trControl = ctrl,
    tuneGrid = expand.grid(cp = c(0.1 , 0.2, 0.3, 0.4)),
    metric = "ROC",
  )
  
  #print model report 
  print(rm_model)
  
  # cp   ROC       Sens       Spec     
  #  0.1  0.554425  0.8762652  0.2325848
  #  0.2  0.500000  0.7600000  0.2400000
  # 0.3  0.500000  0.7600000  0.2400000
  # 0.4  0.500000  0.7600000  0.2400000
  
  #PREDICTING PROBALITY
  y_prob =predict(rm_model , test[-201] ,type='prob')
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob[2] >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  
  # c = 0.1 has highest roc 
  #  0.1  0.554425  0.8762652  0.2325848
  
  ####### test data metric model 2  ##############
  #[1] "accuracy 0.8"
  #[1] "precision 0.17"
  #[1] "recall 0.13"
  #[1] "fpr 0.13"
  #[1] "fnr 0.77"
  #[1] "f1 0.15"
  #Area under the curve:0.554425
  
  #################### observation  model 2   ###########################################################################
  # looking at both base and tuned models it is clear that base model is better than tuned model 
  # base model has higher f1 score 20%  compared to 15%
  # tuned model has higher accuracy from 63% to 80% 
  # tuned model has lower fpr rate   from 35 % to 13%
  # tuned model has higher fnr 47 % to 77% ,this what we are trying to decrease
  # tuned model has lower auc just 55% compared to 59%  (very poor auc )
  
  #coverting target back to 0's and 1's 
  over$Y =ifelse(over$Y =='No',0,1)
  ######################### Manually trying to reduce depth of tree with c =0.1 (model 3) ########################
  #rpart control variable 
  ctrl =rpart.control(cp = 0.01,maxdepth = 5)
  # train model 
  rm_model = rpart( Y~. ,data =over , control = ctrl)
  #predicting probality
  y_prob =predict(rm_model , test[-201] ,type='prob')
  # convert   probality to class according to thresshold (positive class )
  y_pred = ifelse(y_prob[,2] >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  
  #######  test data results  model 3  ########################################################################################
  #[1] "accuracy 0.55" 
  #[1] "precision 0.14"
  #[1] "recall 0.47" 
  #[1] "fpr 0.47" 
  #[1] "fnr 0.33" 
  #[1] "f1 0.21" 
  #Area under the curve: 0.6124
  
  #get auc 
  roc=roc(test[,201], y_prob[,2] )
  print(roc)
  # plot roc_auc curve
  plot(roc ,main="Roc _ auc Decision tree Model  3")
  
  ########################## observation  model 3  ##################################################
  # accuracy of model has decresed from 80% to 55%
  # recall is incresed  47% 
  # fnr has decreased to 33%
  # f1 score has also increased by 1 %
  # This model performs better than base and tunned model still inferior to Logistic Regression  model 
  
  
  
  ################################### Naive Bayes ###############################################
  
  
  ################ using train data without oversampling  (model 1) ###################
  # coverting target to factor 
  train$Y = factor(train$Y ,levels = c(0,1))
  # train model 
  nb_model  =naiveBayes(Y~.  , data =train )  
  
  # model summary  
  
  #PREDICTING PROBALITY
  y_prob =predict(nb_model , test[-201]  ,type='raw')
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob[,2] >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  ################## test data prediction  model 1 ######################################################
  # [1] "accuracy 0.92"
  # [1] "precision 0.72"
  # [1] "recall 0.02"
  # [1] "fpr 0.02"
  # [1] "fnr 0.64"
  # [1] "f1 0.03"
  #Area under the curve: 0.8866
  
  # get Auc 
  roc=roc(test[,201], y_prob[,2] )
  print(roc)
  # plot Roc_Auc curve 
  plot(roc ,main="Roc _ auc  Naive Bayes model 1 ")
  
  
  ##### observation model 1 ###############################################################
  # accuracy of model is good 92%
  #roc_auc is 88% which also good.  
  # but fnr is high 64%  and f1 score very poor only 3%
  # very poor  model performance 
  
  ######################## using over sampled data Model 2 ##################################
  #train model 
  nb_model  =naiveBayes(Y~.  , data =over  )  
  
  #PREDICTING PROBALITY
  y_prob =predict(nb_model , test[-201]  ,type='raw')
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob[,2] >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  
  ######################## test data prediction ################################################
  ############### observation  model 2 #####################################################
  # [1] "accuracy 0.81"      - 
  # [1] "precision 0.32"   -
  # [1] "recall 0.18" +
  # [1] "fpr 0.18" +
  # [1] "fnr 0.21" -
  # [1] "f1 0.23" +
  # Area under the curve: 0.8865
  
  # get auc 
  roc=roc(test[,201], y_prob[,2] )
  print(roc)
  # plot Roc_auc curve 
  plot(roc ,main="Roc _ auc  Naive Bayes model 2 ")
  
  ############ observation model 2    ############################################################

  # accuracy of model has decreased to 81%
  # precision has reduced from 72% to 32%
  # recall for model has increased from  2% to  18%
  # fpr have increased from 2% to 18% 
  # fnr has decreased from 64% to 21 %
  # f1 score has increased from 3% to 23%
  # Auc is same 
  # model has improved but this is also poor model ,still best so far model slightly better than logistic regression
  
#################################################################################
########################## xgboost ##############################################






  # convertion target varible to factor 
  train$Y <- as.numeric(as.factor(train$Y)) - 1 
  test$Y <- as.numeric(as.factor(test$Y)) - 1 
  over$Y <- as.numeric(as.factor(over$Y)) - 1 

  # coverting data into dmatrix as it required in xgboost 
trainD =xgb.DMatrix(data =as.matrix(train[,-201]),label= train$Y)
testD =xgb.DMatrix(data=as.matrix(test[,-201]) ,label  =test$Y)
overD =xgb.DMatrix(data =as.matrix(over[,-201]) ,label=over$Y)

#################### using train data without oversampling #####################
################### model 1 #####################################################

###prameters  used 
# max.depth : max depth tree is allowed to grow
# eta: similar to learing rate 
# nrounds: maximum round algorithmis allowed to run 
# scale_pos_weight: make postive weight 11 times more than neagtive 

#train model  
 xgb1 = xgb.train(
    data = trainD,
    max.depth = 3,
    eta = 0.1,
    nrounds = 500,
    scale_pos_weight =11,
    objective = "binary:logistic"
  )
  
  #PREDICTING PROBALITY
  y_prob =predict(xgb1 , as.matrix(test[,-201] ) )
  # convert   probality to class according to thresshold
  y_pred = ifelse(y_prob >0.5, 1, 0)
  #create confusion matrix 
  conf_matrix= table(test[,201] , y_pred)
  #print model accuracy
  getmodel_accuracy(conf_matrix)
  # get roc 
  roc=roc(test[,201], y_prob )
  print(roc)
  # plot roc
  plot(roc ,main="Roc _ auc  xgboost model 1 ")
########## test data prediction  model 1 #############################################################
  # [1] "accuracy 0.81"
  # [1] "precision 0.32"
  # [1] "recall 0.18"
  # [1] "fpr 0.18"
  # [1] "fnr 0.21"
  # [1] "f1 0.23"
# Area under the curve: 0.8856

########### observation model 1 (xgboost) #######################################  
  # accuracy of model is  81%
  # precision is  32%
  # recall is  18%
  # fpr  is 18% 
  # fnr is 21 %
  # f1 score is  23%
  # Auc is  .8865
  # This model performs allmost same as naive bayes with over sampled dataset
  # only difference is that naive bayes model has sightly high auc by .0009
  

  

###################### using  oversampled data  ###############################
######################### model 2 ##############################################
# train model
xgb1 = xgb.train(
  data = overD,
  max.depth = 3,
  eta = 0.1,
  nrounds = 500,
  scale_pos_weight=2,
  objective = "binary:logistic"
)

#PREDICTING PROBALITY
y_prob =predict(xgb1 , as.matrix(test[,-201] ) )
# convert   probality to class according to thresshold
y_pred = ifelse(y_prob >0.5, 1, 0)
#create confusion matrix 
conf_matrix= table(test[,201] , y_pred)
#print model accuracy
getmodel_accuracy(conf_matrix)
# get roc 
roc=roc(test[,201], y_prob )
print(roc)
# plot roc
plot(roc ,main="Roc _ auc  xgboost model 2 ")
######### test data prediction  model 2 ########################################################
# [1] "accuracy 0.71" 
# [1] "precision 0.24" 
# [1] "recall 0.3" 
# [1] "fpr 0.3" +
# [1] "fnr 0.13" 
# [1] "f1 0.27" 
# Area under the curve: 0.8839


############ observation xgboost (model 2)##########################
# accuracy of model has decreased from  81% to 71% compared to model 1 
# precision of model has decreased from 32% to 24 % compared to model 1
# recall of model has increased  from 18% to 30% compared to model 1
# fpr of model has increased from 18% to 30 % compared to model 1
# fnr of model has decreased from 21% to 13 % compared to model 1
# f1 score  of model has increasd  from 23% to 27 % compared to model 1 
# going by fnr and f1 score model 2 is improvement over  model 1 
# still naive bayes is better 


####################### optimising over sampled data #############################
########################## model 3 ###############################################

# parmeter used 
# gamma :Minimum loss reduction required to make a further partition on a leaf node of the tree.
# early_stopping_rounds : if algo doesnt improve after n round stop 
# print_every_n : print model error after every n round 
#  objective: logistic regression for binary classification, output probability

# watchlist 
wl =list(train =overD, test=testD)
# train model 
xgb1 = xgb.train(
  data = overD,
  max.depth = 3,
  eta = 0.1,
  nrounds = 1000,
  scale_pos_weight=1,
  gamma=1,
  watchlist = wl,
  objective = "binary:logistic",
  early_stopping_rounds = 20,
  print_every_n = 5
    

)

#PREDICTING PROBALITY
y_prob =predict(xgb1 , as.matrix(test[,-201] ) )
# convert   probality to class according to thresshold
y_pred = ifelse(y_prob >0.5, 1, 0)
#create confusion matrix 
conf_matrix= table(test[,201] , y_pred)
#print model accuracy
getmodel_accuracy(conf_matrix)
# get roc 
roc=roc(test[,201], y_prob )
print(roc)
# plot roc
plot(roc ,main="Roc _ auc  xgboost model 3 ")

# test data prediction   model 3 ######################################
# [1] "accuracy 0.86"
# [1] "precision 0.39"
# [1] "recall 0.13"
# [1] "fpr 0.13"
# [1] "fnr 0.26"
# [1] "f1 0.19"
# Area under the curve: 0.8898


############ observation xgboost (model 3)##########################
# accuracy of model has increased from  71% to 86% compared to model 2 
# precision of model has increased from 24% to 39 % compared to model 2
# recall of model has decreased  from 30% to 13% compared to model 2
# fpr of model has decreased from 30% to 13 % compared to model 2
# fnr of model has decreased from 13% to 26 % compared to model 2
# f1 score  of model has decreased  from 27% to 19 % compared to model 2 
# Auc of model3 is higher than model 2 
# going by fnr and f1 score model 3 ,model 2 is still better as it has lower fnr 
# and higher f1 score 
# still naive bayes is better 



  
######################################## optimising  (non over sampled )train data #############################
########################################### model 4 ############################## 
# watch list    
 wl =list(train =overD, test=testD)
#train model 
    xgb6 = xgb.train(
      data = trainD,
      max.depth = 5,
      eta = 0.2,
      nrounds = 1000,
      scale_pos_weight=11,
      gamma=5,
      watchlist = wl,
      objective = "binary:logistic",
      early_stopping_rounds = 20,
      print_every_n = 20
      
      
    )
    
    #PREDICTING PROBALITY
    y_prob =predict(xgb6 , as.matrix(test[,-201] ) )
    # convert   probality to class according to thresshold
    y_pred = ifelse(y_prob >0.5, 1, 0)
    #create confusion matrix 
    conf_matrix= table(test[,201] , y_pred)
    #print model accuracy
    getmodel_accuracy(conf_matrix)
    # get roc 
    roc=roc(test[,201], y_prob )
    print(roc)
    # plot roc
    plot(roc ,main="Roc _ auc  xgboost model 4 ")
 
  ################## test data  prediction model 4  #######################################################
    # [1] "accuracy 0.91"
    # [1] "precision 0.52" 
    # [1] "recall 0.05"
    # [1] "fpr 0.05"-
    # [1] "fnr 0.49" 
    # [1] "f1 0.09"- 
    # Area under the curve: 0.8723

    ############ observation xgboost (model 4 )##################################################################################
    # accuracy of model has increased from  71% to 91% compared to model 2 
    # precision of model has increased from 24% to 52 % compared to model 2
    # recall of model has decreased  from 30% to 5% compared to model 2
    # fpr of model has decreased from 30% to 5 % compared to model 2
    # fnr of model has increased from 13% to 49 % compared to model 2
    # f1 score  of model has decreased  from 27% to 9 % compared to model 2 
    # Auc of model 4 is lower than model 2 
    # going by fnr and f1 score in model 4 ,model 2 is still better as it has lower fnr 
    # and higher f1 score 
    # still naive bayes is better 
    

    

 ########################################################## final model  selection    #######################################################

  # looking at all models with different algorithms 
  # naive bayes model 2  is the best model
  # naive bayes has balanced fnr and fpr
  # f1 score is amongs the highest achieved so far 
  # auc is also over 88%
  
  ### below is model report  of naive bayes model 2 
    # [1] "accuracy 0.81"      - 
    # [1] "precision 0.32"   -
    # [1] "recall 0.18" +
    # [1] "fpr 0.18" +
    # [1] "fnr 0.21" -
    # [1] "f1 0.23" +
    # Area under the curve: 0.8865
    