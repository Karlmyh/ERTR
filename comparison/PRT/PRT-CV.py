# This file can be used to re-generate the results of the probabilstic trees.
# It can also be used to run experiments on new datasets.

import sys
import csv
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
from time import time
import os
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()






data_file_dir = "../../data/real_data_cleaned/"

data_file_name_seq = ['housing_scale.csv', 'mpg_scale.csv','space_ga_scale.csv','mg_scale.csv',
                     'cpusmall_scale.csv','triazines_scale.csv','pyrim_scale.csv',
                      'abalone.csv','bodyfat_scale.csv']


log_file_dir = "../../results/realdata_tree/"


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)

    X = pd.read_csv(data_file_path, header=None, index_col=None)
    X = X.values
    Y = X[:, 0]
    X = scaler.fit_transform(X[:, 1:])


    # Sigma_u validation step, 1e-20 = Std Decision Trees
    sigma_values = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    min_leaf_percentage_values = [0.1]


    
    for k in range(5):

       
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.7 , random_state=k+66)

        
        sigma_Xp = np.std(X_train, axis=0)
        
        parameters = {"splitter":["topk3","topk5"], "criterion":["mseprob"],
                     "min_samples_leaf":[round(len(X_train) * min_leaf_percentage) for min_leaf_percentage in min_leaf_percentage_values],
                     "tol":[sigma_Xp * sigma_val for sigma_val in sigma_values]}
        
        
        cv_model_PRT = GridSearchCV(estimator=tree.DecisionTreeRegressor(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_PRT.fit(X_train, y_train)
        
        time_start = time()
        model_PRT = cv_model_PRT.best_estimator_
        F = [f for f in model_PRT.tree_.feature if f != -2]
        for s_current_node in range(len(F)):
            for k_ind in range(s_current_node + 1, len(F)):
                if F[s_current_node] == F[k_ind]:
                    F[k_ind] = -1
        F = np.array(F)
        prediction = model_PRT.predict3(X_test, F = F)
        error = abs(y_test - prediction)
        MSE_test = np.mean(error ** 2)
        time_end = time()
        
   
        
        log_file_name = "{}.csv".format("PRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          MSE_test, time_end-time_start,
                                          k)
            f.writelines(logs)


