import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import argparse

import sys
import csv
import argparse
from time import time
import os
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


data_file_dir = "../../data/real_data_cleaned/"

data_file_name_seq = ['space_ga_scale.csv','whitewine.csv', 'dakbilgic.csv','mg_scale.csv','bias.csv','cpusmall_scale.csv','aquatic.csv','yacht.csv', 'abalone.csv','cbm.csv']


log_file_dir = "../../results/realdata_forest/"




parser = argparse.ArgumentParser(description='Welcome to TDT builder')

parser.add_argument('-s', type=str, help='splitting method', default='topk3')
parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
parser.add_argument('-t', type=int, help='number of trees', default=50)
parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)
parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)

parser.add_argument('-ts', type=float, help='test size in percentage', default=0.33)
args = parser.parse_args()


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)

    X = pd.read_csv(data_file_path, header=None, index_col=None)
    X = X.values
    Y = X[:, 0]
    X = scaler.fit_transform(X[:, 1:])
    

    sigma = 1

    for k in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k+66)

        rf_predictions = []
        time_start = time()
        for j in range(args.t):
            print('CV=', k, ', T=', j, sep='')

            # Split the dataset with Bootstrap with replacement
            np.random.seed(args.t * k + j)
            idx = np.random.randint(len(X_train), size=len(X_train))
            X_subtree = X_train[idx, :]
            y_subtree = y_train[idx]
            sigma_Xp = np.std(X_subtree, axis=0)

            sigma_Xp = sigma_Xp * sigma
            temp_min_smp_leaf = round(len(X_subtree) * args.l)

            regressor = tree.DecisionTreeRegressor(criterion="mseprob", random_state=0, splitter="topk3",
                                                       min_samples_leaf=temp_min_smp_leaf, tol=sigma_Xp)
            regressor.fit(X_subtree, y_subtree)

            
            F = [f for f in regressor.tree_.feature if f != -2]
            for s_current_node in range(len(F)):
                for kk in range(s_current_node + 1, len(F)):
                    if F[s_current_node] == F[kk]:
                        F[kk] = -1
            F = np.array(F)
            prediction = regressor.predict3(X_test, F=F)

            rf_predictions.append(prediction)

        prediction = np.mean(rf_predictions, axis=0)
        error = abs(y_test - prediction)
        MSE_test = np.mean(error ** 2)
        
        time_end = time()
        
        
        log_file_name = "{}.csv".format("PRRF")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          MSE_test, time_end-time_start,
                                          k)
            f.writelines(logs)



