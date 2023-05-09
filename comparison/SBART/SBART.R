#! /path/to/Rscript

library(readr)
library(tidyverse)
library(caTools)
library(glue)

library(SoftBart)
library(dplyr)


data_file_dir <- "../../data/real_data_cleaned/"


#data_file_name_seq <- c('space_ga_scale.csv','whitewine.csv', "redwine.csv", 'dakbilgic.csv','mg_scale.csv','bias.csv','cpusmall_scale.csv','aquatic.csv','yacht.csv', 'abalone.csv','cbm.csv')

#data_file_name_seq <- c('housing_scale.csv','mpg_scale.csv','music.csv', 'ccpp.csv','concrete.csv','portfolio.csv','building.csv','algerian.csv','fish.csv','communities.csv','forestfires.csv')

data_file_name_seq <- c('algerian.csv' )



repeat_times <- 10

min.max.norm <- function(x){
  ((x-min(x))/(max(x)-min(x)))
}  

# mse 
calculate_error <- function(y,ypredict){
  mse = mean((y-ypredict)**2)
  return(mse)
}
# cross validation 
cv <- function(train_features,train_target,num_tr,tempe,numFold =3){
  error = c(1:numFold)*0
  ind <- sample(1:nrow(train_features),nrow(train_features))
  folds <- cut(seq(1,length(ind)),breaks=numFold,labels=FALSE)
  for (i in c(1:numFold)){
    fold_i = which(folds %in% i)
    index = ind[fold_i]
    train_features_i = as.matrix(train_features[-index,])
    train_target_i = as.matrix(train_target[-index,])
    test_features_i = as.matrix(train_features[index,])
    test_target_i = as.matrix(train_target[index,])
    fit <- softbart(X = train_features_i, Y = train_target_i, X_test = test_features_i, 
                    hypers = Hypers(train_features_i, train_target_i, num_tree = num_tr, temperature = tempe),
                    opts = Opts(num_burn = 500, num_save = 500, update_tau = TRUE))
    
    error[i] = calculate_error(test_target_i,fit$y_hat_test_mean)
  }
  return(mean(error))
}

line_ind<-0
num_cases<- length(data_file_name_seq)*repeat_times
test_errors = data.frame("dataname"=1:num_cases,"Repeat"=1:num_cases,"testerror"=1:num_cases)

# iter for data set
for(data_file_name in data_file_name_seq){
  data_name <- data_file_name
  data_name <- strsplit(data_name, ".", fixed= T)[[1]][1]
  data_file_path = paste(data_file_dir, data_file_name,sep='')
  data <- read_csv(data_file_path,col_names=FALSE)
  X <- data[,2:ncol(data)]
    


  y <- data[,1]
    
   X <- apply(X,2,min.max.norm)
  scaled_data <- cbind(y,X)
    
    
  
  for(i in 1:repeat_times){
    line_ind <- line_ind+1
    set.seed(i+66)
    sub<-sample(1:nrow(scaled_data),round(nrow(scaled_data)*0.8))
    data_train<-scaled_data[sub,]
    data_test<-scaled_data[-sub,]
    X_train<- data_train[,2:ncol(scaled_data)]
    X_test<- data_test[,2:ncol(scaled_data)]
    y_train<- data_train[,1]
    y_test <- data_test[,1]
    train_features = as.matrix(X_train)
    train_target = as.matrix(y_train)
    test_features = as.matrix(X_test)
    test_target = as.matrix(y_test)
      
      
      
   
    # fit with the best
    time_start <- Sys.time()
    fit <- softbart(X = train_features, Y = train_target, X_test = test_features, 
                    hypers = Hypers(train_features, train_target, num_tree = 50, temperature = 1),
                    opts = Opts(num_burn = 2500, num_save = 2500, update_tau = TRUE))
    
    test_error = calculate_error(test_target,fit$y_hat_test_mean)
    time_end <- Sys.time()
    

    log <- sprintf("%s,%f,%f,%s", data_name, test_error, as.numeric(time_end - time_start, units = "secs") , i)
    write(log, "../../results/realdata_forest/SBART.csv",append=TRUE)
  }
  
}










####下面是单个数据集的可以用来调试

# 
# set.seed(123)
# mstereff<-paste(data_file_dir, data_file_name_seq[2],sep='')
# data <- read_csv(mstereff)
# X <- data[,2:ncol(data)]
# X<-X[,-c(43,44)]
# y <- data[,1]
# X <- apply(X,2,min.max.norm)
# data111<- cbind(y,X)
# 
# 
# sub<-sample(1:nrow(data111),round(nrow(data111)*0.8))
# length(sub)
# data_train<-data111[sub,]
# data_test<-data111[-sub,]
# X_train<- data_train[,2:ncol(data111)]
# X_test<- data_test[,2:ncol(data111)]
# y_train<- data_train[[1]]
# y_test <- data_test[[1]]
# train_features = as.matrix(X_train)
# train_target = as.matrix(y_train)
# test_features = as.matrix(X_test)
# test_target = as.matrix(y_test)
# 
# 
# # 构建超参所有可能的组合
# gs <- list(num_tree = c(50, 100),temperature = c(0.1,1,2))%>%cross_df()
# num_combination <- nrow(gs)
# 
# errors = data.frame('num_tree'=1:num_combination,'temperature'=1:num_combination,"Error"=1:num_combination)
# for (i in c(1:num_combination)){
#   temp_ntree = as.numeric(gs[i,1])
#   temp_tempe = as.numeric(gs[i,2])
#   error = cv(train_features,train_target,num_tr=temp_ntree,tempe =temp_tempe, numFold = 5)
#   errors[i,"Error"] = error
#   errors[i,'num_tree'] = temp_ntree
#   errors[i,'temperature'] = temp_tempe
# }
# best_combination_ind<-which(errors$Error==min(errors$Error),arr.ind=TRUE)
# best_combination<-gs[best_combination_ind,]
# # 用最好的再拟合
# fit <- softbart(X = train_features, Y = train_target, X_test = test_features,
#                 hypers = Hypers(train_features, train_target, num_tree = as.numeric(best_combination[1]), temperature = as.numeric(best_combination[2])),
#                 opts = Opts(num_burn = 10, num_save = 10, update_tau = TRUE))
# 
# test_error = calculate_error(test_target,fit$y_hat_test_mean)
# length(test_target)
# 
# 
# 
# 
# 
