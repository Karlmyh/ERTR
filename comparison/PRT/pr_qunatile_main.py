import warnings
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


# Main probabilistic trees (quantile regression)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='dataset path')
    parser.add_argument('-s', type=str, help='splitting method', default='topk3')
    parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
    parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)
    parser.add_argument('-a', type=float, help='alpha value for quantile regression', default=-1.0)
    parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)
    parser.add_argument('-cvs', type=int, help='cross validation min range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

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

    all_errors = []
    for ind, k in enumerate(range(args.cvs, args.cve)):
        print('CV=', k)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)
        sigma_Xp = np.std(X_train, axis=0) * sigmas[ind]
        min_samples_leaf = round(len(X_train) * args.l)

        regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s, samples=None,
                                               min_samples_leaf=min_samples_leaf, tol=sigma_Xp, _alpha=args.a)
        regressor.fit(X_train, y_train)
        F = [f for f in regressor.tree_.feature if f != -2]
        for s_current_node in range(len(F)):
            for kk in range(s_current_node + 1, len(F)):
                if F[s_current_node] == F[kk]:
                    F[kk] = -1
        F = np.array(F)

        prediction = regressor.predict3(X_test, F=F)

        error = []
        for ind, v in enumerate(prediction):
            y_hat = y_test[ind]
            if v < y_hat:
                error.append(float(args.a) * abs(y_hat - v))
            else:
                error.append((1 - float(args.a)) * abs(y_hat - v))
        error = np.array(error)
        mean_error = np.mean(error)
        print(mean_error)
        all_errors.extend(error)

    # Print the Avg RMSE and the Std
    print()
    print('Avg RMSE', np.mean(all_errors))
    print('Std RMSE', np.std(all_errors))
