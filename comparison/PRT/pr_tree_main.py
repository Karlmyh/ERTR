# Probabilstic Trees main file
# This file runs 10 CV WITHOUT validation and it prints the RMSE for each CV and the average,
# This file can be usefull in case the user knows what values to use for sigma_u (noise)

import csv
import pandas as pd
from sklearn import tree
import numpy as np
import cProfile
import time
from sklearn.model_selection import train_test_split
import argparse


# Main probabilistic trees
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='dataset path')
    parser.add_argument('-s', type=str, help='splitting method', default='topk3')
    parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
    parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)
    parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)
    parser.add_argument('-cvs', type=int, help='cross validation min range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()

    # Read the dataset
    X = pd.read_csv(args.x, header=None, index_col=None)
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]

    rmses = []
    sigmas = None
    if args.sg is not None:
        sigmas = [float(v) for v in args.sg.split(',')]
    else:
        sigmas = np.ones(args.cve - args.cvs)

    # 10 Cross-validation, can be customized in the input
    for ind, k in enumerate(range(args.cvs, args.cve)):

        # Split the data into training (80%) and testing (20%), seed is fixed to avoid having different values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)

        # Calculate the Std deviation (noise) with its multiplier
        sigma_Xp = np.std(X_train, axis=0) * sigmas[ind]
        min_samples_leaf = round(len(X_train) * args.l)

        # Run the model
        if args.m == 'mse':
            regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0,
                                                   min_samples_leaf=min_samples_leaf, tol=sigma_Xp)
        else:
            regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s, samples=None,
                                                   min_samples_leaf=min_samples_leaf, tol=sigma_Xp)
        regressor.fit(X_train, y_train)

        # Run the model on the test
        if args.m == 'mse':
            prediction = regressor.predict(X_test)
        else:
            # We need only the features that were used in the construction of the trees (used in the construction of the matrice P)
            F = [f for f in regressor.tree_.feature if f != -2]
            for s_current_node in range(len(F)):
                for k_ind in range(s_current_node + 1, len(F)):
                    if F[s_current_node] == F[k_ind]:
                        F[k_ind] = -1
            F = np.array(F)
            prediction = regressor.predict3(X_test, F=F)

        # Calculate, print and save the RMSE
        error = abs(y_test - prediction)
        RMSE_test = np.sqrt(np.mean(error ** 2))
        print(k, RMSE_test)
        rmses.append(RMSE_test)

    # Print the Avg RMSE and the Std
    print()
    print('Avg RMSE', np.mean(rmses))
    print('Std RMSE', np.std(rmses))
