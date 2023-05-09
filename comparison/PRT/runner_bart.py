import csv
import gc
import sys
import pandas as pd
from sklearn.ensemble import pr_bart as pr_bart
import numpy as np
import time
from sklearn.model_selection import train_test_split
import argparse
import random

# Main uncertain trees
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='x path')
    parser.add_argument('-o', type=str, help='output file')
    parser.add_argument('-e', type=str, help='error file')
    parser.add_argument('-s', type=str, help='splitting method', default='topk3')
    parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
    parser.add_argument('-t', type=str, help='type of smoothing', default='smoothed')
    parser.add_argument('-n', type=int, help='number of estimator', default=100)
    parser.add_argument('-i', type=int, help='number of burn-in iteration', default=50)
    parser.add_argument('-b', type=int, help='number of after burn-in iteration', default=200)
    parser.add_argument('-l', type=float, help='min leaf percentage', default=0.05)
    parser.add_argument('-d', type=int, help='max depth', default=None)
    parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)
    parser.add_argument('-cvs', type=int, help='cross validation min/max range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()

    X = pd.read_csv(args.x, header=None, index_col=None)
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]

    outputs = []
    times = []
    rmses = []
    all_errors = []
    all_predictions = []
    all_y = []

    sigmas = None
    if args.sg is not None:
        sigmas = [float(v) for v in args.sg.split(',')]
    else:
        sigmas = np.ones(args.cve - args.cvs)

    yMaxMinHalved = 0.5

    for ind, k in enumerate(range(args.cvs, args.cve)):

        max_RMSE = sys.float_info.max
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)

        ############################################################################################
        # Normalizing Y to be in (-0.5, 0.5)
        min_y = np.min(y_train)
        max_y = np.max(y_train)
        subtraction = max_y - min_y
        y_train -= min_y
        y_train = np.array([x / subtraction for x in y_train], dtype=np.float64)
        y_train -= yMaxMinHalved
        ############################################################################################

        print('Cross Validation:', k)
        min_samples_leaf = round(len(X_train) * args.l)
        random_seed = random.randint(0, 10000)
        print('Random seed', random_seed)
        regressor = pr_bart.BARTRegressor(random_state=random_seed, n_estimators=args.n,
                                          presort=False, min_samples_leaf=min_samples_leaf,
                                          sigma_Xp=sigmas[ind], sigma_type='std',
                                          n_iteration=args.i, n_after_burn_iteration=args.b, criterion=args.m)
        t = time.time()
        regressor.fit(X_train, y_train)
        t = time.time() - t
        print('Time', t)

        t = time.time()
        prediction = regressor.predict(X_test)
        t = time.time() - t
        print('Predict time', t)
        ##########################################################################################
        prediction += yMaxMinHalved
        prediction *= subtraction
        prediction += min_y
        ##########################################################################################
        all_predictions.extend(prediction)
        all_y.extend(y_test)
        error = abs(y_test - prediction)
        all_errors.extend(error)
        RMSE_test = np.sqrt(np.mean(error ** 2))
        print('RMSE', RMSE_test)
        avg_nb_nodes = 0
        for tree_sampler in regressor.estimators_samplers:
            for tree in tree_sampler:
                avg_nb_nodes += tree.tree_.node_count

        avg_nb_nodes /= (args.n * args.b)
        print('Avg Nb Nodes', avg_nb_nodes)
        outputs.append([k, RMSE_test, t, sigmas[ind], random_seed])

        times.append(t)
        rmses.append(RMSE_test)

        with open(args.o, 'w') as csv_file:
            spam_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for output in outputs:
                spam_writer.writerow(output)

    print(rmses)
    all_errors = np.array(all_errors)
    outputs.append(['Average MSE', np.mean(rmses)])
    outputs.append(['STD', np.std(rmses)])
    print('Average MSE', np.mean(rmses))
    print('Average Std', np.std(rmses))
    outputs.append(['Average Time', np.mean(times)])
    outputs.append(['Total Average MSE', np.sqrt(np.mean(all_errors ** 2))])
    gc.collect()

    with open(args.o, 'w') as csv_file:
        spam_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for output in outputs:
            spam_writer.writerow(output)

    all_errors = all_errors.T
    pd.DataFrame(all_errors).to_csv(args.e, index=False, header=False)
