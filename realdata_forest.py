import os
import numpy as np 
import pandas as pd
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE

from RTER import RegressionTree
from ensemble import RegressionTreeBoosting, RegressionTreeEnsemble



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor






data_file_dir = "./data/real_data_cleaned/"

#data_file_name_seq = ['space_ga_scale.csv','whitewine.csv', 'dakbilgic.csv','bias.csv', 'cpusmall_scale.csv', 'airfoil.csv', 'mg_scale.csv', 'abalone.csv','cbm.csv', "algerian.csv","ccpp.csv","forestfires.csv","redwine.csv"]

data_file_name_seq = ['space_ga_scale.csv','whitewine.csv', 'dakbilgic.csv','bias.csv', 'cpusmall_scale.csv', 'airfoil.csv', 'mg_scale.csv', 'abalone.csv','cbm.csv', "algerian.csv","ccpp.csv","forestfires.csv","redwine.csv"]


#data_seq = glob.glob("{}/*.csv".format(log_file_dir))
#data_file_name_seq = [os.path.split(data)[1] for data in data_seq]

log_file_dir = "./results/realdata_forest/"


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)
    data = pd.read_csv(data_file_path,header = None)
    data = np.array(data)
    
    X = data[:,1:]
    y = data[:,0]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    

    repeat_times = 3
        
    for i in range(repeat_times):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+256)
        
        
        # RTER ensemble
      
        parameters={"n_estimators":[200], "max_features":[0.5,0.75,1],
                    "max_samples":[0.8,1,1.2],
       "min_samples_split":[5], "max_depth":[5,6,7,8,9],
       "order":[0],"splitter":["varreduction"],
        "estimator":["pointwise_extrapolation_estimator"],
       "r_range_low":[0],"r_range_up":[1],
       "lamda":[0.0001,0.001,0.01],"V":[2,"auto"]}
        cv_model_ensemble=GridSearchCV(estimator=RegressionTreeEnsemble(),param_grid=parameters, cv=3, n_jobs=50)
        
        cv_model_ensemble.fit(X_train, y_train)
        ensemble_model = cv_model_ensemble.best_estimator_
        
        time_start=time()
        ensemble_model.ensemble_parallel = 1
        mse_score= - ensemble_model.score(X_test, y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("RTER-RF")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
        '''
          
        parameters = {"n_estimators":[200],"max_depth":[2,4,6,8]}
        
        cv_model_RFR = GridSearchCV(estimator=RandomForestRegressor(),param_grid=parameters, cv=3, n_jobs=-1) 
        cv_model_RFR.fit(X_train, y_train)
        model_RFR = cv_model_RFR.best_estimator_
        
        time_start=time()
        model_RFR.fit(X_train, y_train)
        prediction = model_RFR.predict(X_test)
        mse_score = np.mean((prediction - y_test)**2)
        time_end=time()
        
        log_file_name = "{}.csv".format("RF")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
         

        '''
    
        