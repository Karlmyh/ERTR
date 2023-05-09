import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse


# Main uncertain trees
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='x path')
    parser.add_argument('-y', type=str, help='y path')
    parser.add_argument('-f', type=str, help='name')
    parser.add_argument('-cvs', type=int, help='cross validation min/max range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()

    X = pd.read_csv(args.x, header=None, index_col=None)

    # Convert pandas into nd array
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]

    folder = 'soft-tree/' + args.f
    os.mkdir(folder)

    for k in range(args.cvs, args.cve):
        m = (k / 2) + 1
        n = (k % 2) + 1
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)

        X_valid_tr, X_valid_test, y_valid_tr, y_valid_test = train_test_split(X_train, y_train,
                                                                              test_size=args.ts, random_state=0)

        new_train = np.c_[X_valid_tr, y_valid_tr]
        new_valid = np.c_[X_valid_test, y_valid_test]
        new_test = np.c_[X_test, y_test]

        pd.DataFrame(new_train).to_csv(folder + '/' + args.f + '-train-' + str(k) + '.txt',
                                       index=False, header=False, sep=' ')
        pd.DataFrame(new_valid).to_csv(folder + '/' + args.f + '-validation-' + str(k) + '.txt',
                                       index=False, header=False, sep=' ')
        pd.DataFrame(new_test).to_csv(folder + '/' + args.f + '-test-' + str(k) + '.txt',
                                      index=False, header=False, sep=' ')
