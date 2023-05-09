#! /path/to/Rscript



library(readr)
library(tidyverse)
library(caTools)
library(glue)

library(BooST)
library(dplyr)


data_file_dir <- "../../data/real_data_cleaned/"

data_file_name_seq <- c('mpg_scale.csv','airfoil.csv','space_ga_scale.csv','whitewine.csv', 'dakbilgic.csv','mg_scale.csv','bias.csv','cpusmall_scale.csv','aquatic.csv','music.csv', 'redwine.csv','ccpp.csv','concrete.csv','portfolio.csv','building.csv','yacht.csv', 'abalone.csv','facebook.csv','algerian.csv','fish.csv','communities.csv','forestfires.csv','cbm.csv','housing_scale.csv')


repeat_times <- 10

min.max.norm <- function(x){
  ((x-min(x))/(max(x)-min(x)))
}  

# evaluate tree
eval_tree=function(x,tree){
  terminal=tree[which(tree$terminal=="yes"),]
  logimat=matrix(NA,nrow(x),nrow(terminal))
  for(i in 1:nrow(terminal)){
    node=terminal[i,]
    logit=1/(1+exp(-node$gamma*(x[,node$variable]-node$c0)))
    if(node$side==2){logit=1-logit}
    parent=node$parent
    while(parent!=0){
      node=tree[parent,]
      logitaux=1/(1+exp(-node$gamma*(x[,node$variable]-node$c0)))
      if(node$side==2){logitaux=1-logitaux}
      logit=logit*logitaux
      parent=node$parent
    }
    logimat[,i]=logit
  }
  
  fitted=logimat%*%terminal$b
  return(fitted)
}

eval_boost <- function(newx,object){
  
  v=object$v
  y0=object$ybar
  rho=object$rho
  model=object$Model
  lr=rho*v
  
  fitaux=t(t(Reduce("cbind",lapply(model,function(t)eval_tree(newx,t$tree))))*lr)
  fitted.values=y0 + rowSums(fitaux)
  return(fitted.values)
}

# mse
calculate_error <- function(y,ypredict){
  mse = mean((y-ypredict)**2)
  return(mse)
}
# cross validation
cv <- function(train_features,train_target,v,d_max,M,numFold =5){
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
    BooST_obj <- BooST(x = train_features_i, y = train_target_i, v = v, d_max = d_max,M=M)
    
    error[i] = calculate_error(test_target_i,eval_boost( test_features_i,BooST_obj ))
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
  data <- read_csv(data_file_path,col_names = FALSE)
  X <- data[,2:ncol(data)]
  
  y <- data[,1]
  

  if (data_name == "cbm"){
    X[,8]<- X[,8] + rnorm(nrow(X),0,1)
    X[,11]<- X[,11] + rnorm(nrow(X),0,0.1)
  }
  
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
    
    
    time_start <- Sys.time()
    best_BooST_obj <- BooST(x = train_features, y = train_target, d_max = 4,M = 100 )
      
      print(test_target[1:10])
      print(eval_boost( test_features,best_BooST_obj )[1:10])
    
    test_error = calculate_error(test_target,eval_boost( test_features,best_BooST_obj ))
    time_end <- Sys.time()
    
    
    log <- sprintf("%s,%f,%f,%s", data_name, test_error, as.numeric(time_end - time_start,units = "secs") , i)
    write(log, "../../results/realdata_boosting/STRB.csv",append=TRUE)
  }
  
}








