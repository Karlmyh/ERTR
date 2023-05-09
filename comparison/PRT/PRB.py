import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import argparse
from time import time
import os

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


data_file_dir = "../../data/real_data_cleaned/"

data_file_name_seq = ['airfoil.csv', 'mg_scale.csv', 'abalone.csv','space_ga_scale.csv', "algerian.csv","forestfires.csv","redwine.csv",'whitewine.csv', 'dakbilgic.csv','bias.csv', 'cpusmall_scale.csv', 'cbm.csv',"ccpp.csv"]

# ['housing_scale.csv','mpg_scale.csv','music.csv', 'ccpp.csv','concrete.csv','portfolio.csv','building.csv','algerian.csv','fish.csv','communities.csv','forestfires.csv']


log_file_dir = "../../results/realdata_boosting/"






parser = argparse.ArgumentParser(description='Welcome to TDT builder')
parser.add_argument('-x', type=str, help='dataset path')
parser.add_argument('-s', type=str, help='splitting method', default='topv3')
parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
parser.add_argument('-t', type=int, help='number of trees', default=100)
parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)
parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)
parser.add_argument('-cvs', type=int, help='cross validation min range', default=0)
parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
parser.add_argument('-ts', type=float, help='test size in percentage', default=0.3)
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


    
        print('CV=', k)
        
        min_samples_leaf = round(len(X_train) * args.l)

        # Presort needs to be False (for now)
        
        time_start = time()
        regressor = ensemble.GradientBoostingRegressor(random_state=0, n_estimators=args.t, presort=False,
                                                       min_samples_leaf=min_samples_leaf, criterion=args.m)
        
        
        regressor.fit_un(X_train, y_train, sigma_mult=  1 )


        prediction = regressor.predict(X_test)

        error = abs(y_test - prediction)
        MSE_test = np.mean(error ** 2)
        
        time_end = time()
        
        
        log_file_name = "{}.csv".format("PRB")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          MSE_test, time_end-time_start,
                                          k)
            f.writelines(logs)


