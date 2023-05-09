import os
import numpy as np 
import pandas as pd
import glob

# accuracy summarize
log_file_dir = "../accuracy"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "distribution,mse,time,seed,n_train,n_test".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["method"],columns=["distribution"], values=[ "mse"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./accuracy_summary.xlsx")



# efficiency summarize
log_file_dir = "../efficiency"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "distribution,time,seed,n_train,n_test".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["method","n_train"],columns=["distribution","n_test"], values=[ "time"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./efficiency_summary.xlsx")

