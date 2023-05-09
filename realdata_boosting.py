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

#data_file_name_seq = ['airfoil.csv', 'mg_scale.csv', 'abalone.csv','space_ga_scale.csv', "algerian.csv","forestfires.csv","redwine.csv",'whitewine.csv', 'dakbilgic.csv','bias.csv', 'cpusmall_scale.csv', 'cbm.csv',"ccpp.csv"]

data_file_name_seq = ["forestfires.csv"]

#data_seq = glob.glob("{}/*.csv".format(log_file_dir))
#data_file_name_seq = [os.path.split(data)[1] for data in data_seq]

log_file_dir = "./results/realdata_boosting/"


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
    

    repeat_times = 10
        
    for i in range(19,repeat_times+10):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+256)
        

        # RTER with var reduction
        parameters={"min_samples_split":[2,5], "max_depth":[4,5,6,7],
   "order":[0],"splitter":["varreduction"],
    "estimator":["pointwise_extrapolation_estimator"],
   "r_range_low":[0],"r_range_up":[1],
   "lamda":[0.001,0.01],"V":["auto"],
               "n_estimators":[100], "max_features":[1,0.7],
               "max_samples":[1],"rho":[0.05,0.1]}
        
        
        # RTER boosting
        
        cv_model_boosting=GridSearchCV(estimator=RegressionTreeBoosting(),param_grid=parameters, cv=3, n_jobs=64)
        cv_model_boosting.fit(X_train, y_train)
        time_start=time()
        boosting_model = cv_model_boosting.best_estimator_
        mse_score= - boosting_model.score(X_test, y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("RTER-boosting")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
            
            
        gbrtparameters = {"n_estimators":[100],"max_depth":[2,4,6,8]}
        
        cv_model_GBRT = GridSearchCV(estimator=GradientBoostingRegressor(),param_grid=gbrtparameters, cv=3, n_jobs=-1) 
        cv_model_GBRT.fit(X_train, y_train)
        model_GBRT = cv_model_GBRT.best_estimator_
        
        time_start=time()
        model_GBRT.fit(X_train, y_train)
        prediction = model_GBRT.predict(X_test)
        mse_score = np.mean((prediction - y_test)**2)
        time_end=time()
        
        log_file_name = "{}.csv".format("GBRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
         
       
     
    
        