# This file can be used to re-generate the results of the probabilstic trees.
# It can also be used to run experiments on new datasets.

import sys
import csv
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from time import time
import os
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Validator probabilistic trees


parser = argparse.ArgumentParser(description='Welcome to TDT builder')
parser.add_argument('-s', type=str, help='splitting method', default='topk3')
parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
#parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)    
parser.add_argument('-cvs', type=int, help='cross validation min/max range', default=0)
parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
parser.add_argument('-ts', type=float, help='test size in percentage', default=0.3)
args = parser.parse_args()


data_file_dir = "../../data/real_data_cleaned/"

'''

data_file_name_seq = ['housing_scale.csv',
 'mpg_scale.csv',
 'airfoil.csv',
 'space_ga_scale.csv',
 'whitewine.csv',
 'dakbilgic.csv',
 'mg_scale.csv',
 'bias.csv',
 'cpusmall_scale.csv',                     
 'aquatic.csv',
 'music.csv',
 'redwine.csv',
 'ccpp.csv',
 'concrete.csv',
 'portfolio.csv',
 'building.csv',
 'yacht.csv',
 'abalone.csv',
 'facebook.csv',
 'algerian.csv',
 'fish.csv',
 'communities.csv',
 'forestfires.csv',
 'cbm.csv']

'''

data_file_name_seq = [
 'housing_scale.csv',
 'mpg_scale.csv',
 'airfoil.csv',
 'space_ga_scale.csv',
 'whitewine.csv',
 'dakbilgic.csv',
 'mg_scale.csv',
 'bias.csv',
 'cpusmall_scale.csv',                     
 'aquatic.csv',
 'music.csv',
 'redwine.csv',
 'ccpp.csv',
 'concrete.csv',
 'portfolio.csv',
 'building.csv',
 'yacht.csv',
 'abalone.csv',
 'algerian.csv',
 'fish.csv',
 'communities.csv',
 'forestfires.csv',
 'cbm.csv']

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


    # 10 Cross-validation by default, can be changed by the user in the input.
    for k in range(20):

        # Split the dataset into training (70%) and testing (30%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k+66)

        # Calculate the standard deviation (noise)
        sigma_Xp = np.std(X_train, axis=0)
        min_error = sys.float_info.max
        errors_arr = []
        regressor_arr = []

        # Split the training (70%) into a validation training set (~49% of the original size) 
        # and validation testing set (~21% of the original size)
        # Seed is fixed to avoid different values
        X_tr_valid, X_ts_valid, y_tr_valid, y_ts_valid = train_test_split(X_train, y_train,
                                                                          test_size=args.ts, random_state=0)

        # The stopping criteria requires all leaves to have at least 10% of the training size


        # Validation loop
        for sigma_val in sigma_values:
            for min_leaf_percentage in min_leaf_percentage_values:

                temp_min_smp_leaf = round(len(X_tr_valid) * min_leaf_percentage)

            # Calculate the new standard deviation based on the noise modifier
                sigma_arr = sigma_Xp * sigma_val

                # Run the model
                regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s,
                                                           min_samples_leaf=temp_min_smp_leaf, tol=sigma_arr)
                regressor.fit(X_tr_valid, y_tr_valid)

                # Test the model on the validation test set
                F = [f for f in regressor.tree_.feature if f != -2]
                for s_current_node in range(len(F)):
                    for k_ind in range(s_current_node + 1, len(F)):
                        if F[s_current_node] == F[k_ind]:
                            F[k_ind] = -1
                F = np.array(F)
                prediction = regressor.predict3(X_ts_valid, F=F)

                # Calculate the MSE
                error = abs(y_ts_valid - prediction)
                MSE_test = np.mean(error ** 2)

                # print the current CV, current sigma_modifier, and current MSE
                print(k, sigma_val,min_leaf_percentage, "{:.3f}".format(MSE_test))
                errors_arr.append(MSE_test)
                regressor_arr.append(regressor)

        # Testing
        # Pick the best value of sigma_u (standard deviation) based on the validation
        ranking_sigma = np.argsort(errors_arr)
        best_sigma = sigma_values[ranking_sigma[0] // len(min_leaf_percentage_values)]
        best_min_leaf_percentage = sigma_values[ranking_sigma[0] % len(min_leaf_percentage_values)]
        sigma_arr = sigma_Xp * best_sigma

        # Recalculate the stopping criteria
        temp_min_smp_leaf = round(len(X_train) * best_min_leaf_percentage)

        # Run the model on the training set
        time_start = time()
        regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s,
                                                   min_samples_leaf=temp_min_smp_leaf, tol=sigma_arr)
        regressor.fit(X_train, y_train)

        # Run the model on the test set
        F = [f for f in regressor.tree_.feature if f != -2]
        for s_current_node in range(len(F)):
            for k_ind in range(s_current_node + 1, len(F)):
                if F[s_current_node] == F[k_ind]:
                    F[k_ind] = -1
        F = np.array(F)
        prediction = regressor.predict3(X_test, F=F)
        error = abs(y_test - prediction)
        MSE_test = np.mean(error ** 2)
        
        time_end = time()

        # Print the best MSE
        print('Best', k, best_sigma, best_min_leaf_percentage, "{:.3f}".format(MSE_test))
        
        
        log_file_name = "{}.csv".format("PRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          MSE_test, time_end-time_start,
                                          k)
            f.writelines(logs)


