import numpy as np
from time import time
import os

from distribution import TestDistribution
from RTER import RegressionTree
from ensemble import RegressionTreeBoosting, RegressionTreeEnsemble


from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor





distribution_index_vec=[1,2,3,4,5,6,7,8,9,10]
repeat_time=5


log_file_dir = "./results/performace/"


for distribution_iter,distribution_index in enumerate(distribution_index_vec):

    for iterate in range(repeat_time):


        
        np.random.seed(iterate+256)
        # generate distribution


        sample_generator=TestDistribution(distribution_index).returnDistribution()
        n_test, n_train = 5000,1000
        X_train, Y_train = sample_generator.generate(n_train)
        X_test, Y_test = sample_generator.generate(n_test)
        
        
        # single tree 
        parameters={"min_samples_split":[5,15], "max_depth":[0,1,2,3,4,5,6,7],
           "splitter":["maxedge"],"estimator":["naive_estimator"]}
        
        cv_model_tree=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=3, n_jobs=-1)
        cv_model_tree.fit(X_train, Y_train)
        tree_model = cv_model_tree.best_estimator_
        
        time_start=time()
        tree_model.fit(X_train, Y_train)
        mse_score=-tree_model.score(X_test, Y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("tree")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)

        
        # RTER 
        parameters={"min_samples_split":[10,30], "max_depth":[1,2,3,4,5],
           "order":[0,1,2,3,6],"splitter":["maxedge"],
            "estimator":["pointwise_extrapolation_estimator"],
           "r_range_low":[0,0.1],"r_range_up":[0.6,0.8,1],
           "lamda":[0.0001,0.001,0.01,0.1,1,5],"V":[3,5,7,9,12,15,20]}
        
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=3, n_jobs=-1)
        cv_model_RTER.fit(X_train, Y_train)
        RTER_model = cv_model_RTER.best_estimator_
        RTER_model.parallel_jobs = "auto"
        
        time_start=time()
        RTER_model.fit(X_train, Y_train)
        mse_score=-RTER_model.score(X_test, Y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("RTER")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
        # boosting
        parameters= {"min_samples_split":[10], "max_depth":[1,2,3,4,5],
                     "splitter":["maxedge"], "estimator":["naive_estimator"],
                     "n_estimators":[100,150,200,300],"rho":[0.1]}
        
        cv_model_boosting=GridSearchCV(estimator=RegressionTreeBoosting(),param_grid=parameters, cv=10, n_jobs=-1)
        cv_model_boosting.fit(X_train, Y_train)
        boosting_model = cv_model_boosting.best_estimator_
        
        time_start=time()
        boosting_model.fit(X_train, Y_train)
        mse_score= - boosting_model.score(X_test, Y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("boosting")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
            
        # ensemble
        parameters= {"min_samples_split":[10], "max_depth":[1,2,3,4,5],
                     "splitter":["maxedge"], "estimator":["naive_estimator"],
                     "n_estimators":[200,300,400,700]}
        
        cv_model_ensemble=GridSearchCV(estimator=RegressionTreeEnsemble(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_ensemble.fit(X_train, Y_train)
        ensemble_model = cv_model_ensemble.best_estimator_
        
        time_start=time()
        ensemble_model.ensemble_parallel = 1
        ensemble_model.fit(X_train, Y_train)
        mse_score= - ensemble_model.score(X_test, Y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("ensemble")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
        
     
        # GBRT
        parameters= {"n_estimators":[500,1000,2000], "learning_rate":[0.01,0.05]}
        
        cv_model_GBRT=GridSearchCV(estimator=GradientBoostingRegressor(),param_grid=parameters, cv=10, n_jobs=-1)
        cv_model_GBRT.fit(X_train, Y_train)
        model_GBRT = cv_model_GBRT.best_estimator_
        
        time_start=time()
        model_GBRT.fit(X_train, Y_train)
        mse_score = model_GBRT.score(X_test,Y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("GBRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
            
        # RF
        
        parameters = {"n_estimators":[10,100,200]}
        
        cv_model_RFR = GridSearchCV(estimator=RandomForestRegressor(),param_grid=parameters, cv=10, n_jobs=-1) 
        cv_model_RFR.fit(X_train, Y_train)
        model_RFR = cv_model_RFR.best_estimator_
        
        time_start=time()
        model_RFR.fit(X_train, Y_train)
        mse_score = model_RFR.score(X_test,Y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("RFR")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
            
        # DT
        
        parameters = {"max_depth":[2,5,8]}
        
        cv_model_DT = GridSearchCV(estimator=DecisionTreeRegressor(),param_grid=parameters, cv=5, n_jobs=-1) 
        cv_model_DT.fit(X_train, Y_train)
        model_DT = cv_model_DT.best_estimator_
        
        time_start=time()
        model_DT.fit(X_train, Y_train)
        mse_score=model_DT.score(X_test)
        time_end=time()
    
        log_file_name = "{}.csv".format("DT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
      