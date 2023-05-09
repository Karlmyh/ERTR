import os
import numpy as np 
import pandas as pd
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE

from PRRT import RegressionTree



from sklearn.tree import DecisionTreeRegressor





data_file_dir = "../data/real_data_cleaned/"

data_file_name_seq = ['space_ga_scale.csv','whitewine.csv', 'dakbilgic.csv','bias.csv', 'cpusmall_scale.csv', 'airfoil.csv', 'mg_scale.csv', 'abalone.csv','cbm.csv', "algerian.csv","ccpp.csv","forestfires.csv","redwine.csv"]



log_file_dir = "../results/realdata_random_tree/"


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)
    data = pd.read_csv(data_file_path, header = None)
    data = np.array(data)
    
    X = data[:,1:]
    y = data[:,0]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    

    repeat_times = 20
        
    for i in range(repeat_times):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+66)
        
        
        # PRT 
        parameters={"min_samples_split":[2], "max_depth":[2,4,6,8],"splitter":["maxedge"],
        "estimator":["pr_estimator"],"lamda":[0.25,0.5,0.75,1,1.25,1.5,1.75,2]}
        
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=3, n_jobs=50)
        cv_model_RTER.fit(X_train, y_train)
        
        
        time_start=time()
        RTER_model = cv_model_RTER.best_estimator_
        RTER_model.parallel_jobs = "auto"
        mse_score= -RTER_model.score(X_test, y_test)
        time_end=time()
     
        log_file_name = "{}.csv".format("PRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
       

       
           

            
     
    
        