# Toy example to test Probabilistic trees

import pandas as pd
import numpy as np
import argparse
from sklearn import tree
from sklearn.model_selection import train_test_split


# Toy example probabilistic trees
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='x path')
    parser.add_argument('-s', type=str, help='splitting method (topv1, topv3, topv5)', default='topv3')
    parser.add_argument('-m', type=str, help='criterion method (mseprob for probabilstic_tree), '
                                             'mse for standard trees', default='mseprob')
    parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()
	
	# Read the dataset
    X = pd.read_csv(args.x, header=None, index_col=None)
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]
	
	# Split the dataset into training (80%) and test (20%) of the total size, the seed is fixed here
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=42)
	
	# Calculate the standard deviation of X
    sigma_Xp = np.std(X_train, axis=0)
	
	# Stopping criteria requires all regions/leaves to have at least 10% of the training data
    min_samples_leaf = round(len(X_train) * args.l)
	
	# Build the model
    if args.m in ['mse']:
        regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0,
                                               min_samples_leaf=min_samples_leaf, tol=sigma_Xp)
    else:
        regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s,
                                               min_samples_leaf=min_samples_leaf, tol=sigma_Xp)
    regressor.fit(X_train, y_train)
	
	# Running the model on the test data
    if args.m == 'mse':
        prediction = regressor.predict(X_test)
    else:
        # We only the features that are used in the construction of the tree
        F = [f for f in regressor.tree_.feature if f != -2]
        for s_current_node in range(len(F)):
            for k_ind in range(s_current_node + 1, len(F)):
                if F[s_current_node] == F[k_ind]:
                    F[k_ind] = -1
        F = np.array(F)
        prediction = regressor.predict3(X_test, F=F)
	
	# Calculate and print the RMSE
    error = abs(y_test - prediction)
    RMSE_test = np.sqrt(np.mean(error ** 2))
    print('RMSE:', RMSE_test)
