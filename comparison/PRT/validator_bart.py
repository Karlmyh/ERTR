import csv
import gc
import sys
import pandas as pd
import random
from sklearn.ensemble import pr_bart as pr_bart
import numpy as np
import time
from sklearn.model_selection import train_test_split
import argparse


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
    parser.add_argument('-r', type=str, help='delimeter', default=',')
    parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)
    parser.add_argument('-cvs', type=int, help='cross validation min/max range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()

    X = pd.read_csv(args.x, header=None, index_col=None)

    # Convert pandas into nd array
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]
    # X = quantile_normalization(X)

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

        best_sigma = 1
        max_RMSE = sys.float_info.max
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)

        ##################################################################
        # Normalizing Y to be in (-0.5, 0.5)
        min_y = np.min(y_train)
        max_y = np.max(y_train)
        subtraction = max_y - min_y

        y_train -= min_y
        y_train = np.array([x / subtraction for x in y_train], dtype=np.float64)
        y_train -= yMaxMinHalved

        ##################################################################

        print('Cross Validation:', k)

        X_tr_valid, X_ts_valid, y_tr_valid, y_ts_valid = train_test_split(X_train, y_train,
                                                                          test_size=args.ts, random_state=0)
        random_seed = random.randint(0, 10000)
        try:
            min_samples_leaf = round(len(X_tr_valid) * args.l)
            for sigma in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
                y_ts_valid_copy = np.copy(y_ts_valid)
                print('Validation with sigma = %f and random seed' % sigma, random_seed)

                valid_n_iter = int(args.i / 5)
                valid_b_iter = int(args.b / 5)
                regressor = pr_bart.BARTRegressor(random_state=random_seed, n_estimators=args.n,
                                                  presort=False,  min_samples_leaf=min_samples_leaf,
                                                  sigma_Xp=sigma, n_iteration=valid_n_iter, sigma_type='std',
                                                  n_after_burn_iteration=valid_b_iter, criterion=args.m)
                regressor.fit(X_tr_valid, y_tr_valid)
                prediction = regressor.predict(X_ts_valid)

                ################################################################
                prediction += yMaxMinHalved
                prediction *= subtraction
                prediction += min_y
                y_ts_valid_copy += yMaxMinHalved
                y_ts_valid_copy *= subtraction
                y_ts_valid_copy += min_y
                ################################################################
                error = abs(y_ts_valid_copy - prediction)
                RMSE_test = np.sqrt(np.mean(error ** 2))
                print('Valid RMSE_test', RMSE_test)
                if RMSE_test < max_RMSE:
                    max_RMSE = RMSE_test
                    best_sigma = sigma

        except Exception as e:
            print('*************************************')
            print('Error with K= ', str(e))
            print('*************************************')
        print('Training started with best sigma %f' % best_sigma)

        try:
            min_samples_leaf = round(len(X_train) * args.l)
            regressor = pr_bart.BARTRegressor(random_state=random_seed, n_estimators=args.n, presort=False,
                                              sigma_Xp=best_sigma, n_iteration=args.i, criterion=args.m,
                                              sigma_type='std',
                                              n_after_burn_iteration=args.b, min_samples_leaf=min_samples_leaf)
            t = time.process_time()
            regressor.fit(X_train, y_train)
            t = time.process_time() - t

            prediction = regressor.predict(X_test)

            ################################################################
            prediction += yMaxMinHalved
            prediction *= subtraction
            prediction += min_y
            ################################################################
            all_predictions.extend(prediction)
            all_y.extend(y_test)
            error = abs(y_test - prediction)
            all_errors.extend(error)
            RMSE_test = np.sqrt(np.mean(error ** 2))
            print(RMSE_test)
            outputs.append([k, RMSE_test, t, best_sigma, random_seed])

            times.append(t)
            rmses.append(RMSE_test)

            with open(args.o, 'w') as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for output in outputs:
                    spam_writer.writerow(output)

        except Exception as e:
            print('*************************************')
            print('Error with K= ', str(e))
            print('*************************************')

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
