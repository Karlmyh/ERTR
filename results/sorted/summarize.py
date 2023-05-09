import os
import numpy as np 
import pandas as pd
import glob
from scipy.stats import wilcoxon




report_dataset = ['abalone',"airfoil","algerian",'space_ga_scale',"ccpp",'whitewine', 'dakbilgic','mg_scale','bias','cpusmall_scale','forestfires', "redwine",'cbm']

# real data tree summarize
log_file_dir = "../realdata_tree"

ST_table = pd.read_csv("../realdata_tree/ST.txt",header = None)
ST_table.to_csv("../realdata_tree/ST.csv", header = None, index = None)


method_seq = glob.glob("{}/*.csv".format(log_file_dir))
method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)
summarize_log=pd.DataFrame([])

for method in method_seq:
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    log.columns = "dataset,mse,time,iteration".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    

summarize_log = summarize_log[summarize_log["dataset"].isin(report_dataset)]
    
summary = pd.pivot_table(summarize_log, index=["dataset"],columns=["method"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])
summary.to_excel("./realdata_tree_summary.xlsx")


# real data random tree summarize
log_file_dir = "../realdata_random_tree"


method_seq = glob.glob("{}/*.csv".format(log_file_dir))
method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)
summarize_log=pd.DataFrame([])

for method in method_seq:
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    log.columns = "dataset,mse,time,iteration".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    

summarize_log = summarize_log[summarize_log["dataset"].isin(report_dataset)]
    
summary = pd.pivot_table(summarize_log, index=["dataset"],columns=["method"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])
summary.to_excel("./realdata_random_tree_summary.xlsx")





# real data forest summarize
log_file_dir = "../realdata_forest"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))
method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)
summarize_log=pd.DataFrame([])

for method in method_seq:
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    log.columns = "dataset,mse,time,iteration".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)

summarize_log = summarize_log[summarize_log["dataset"].isin(report_dataset)]
    
    
summary = pd.pivot_table(summarize_log, index=["dataset"],columns=["method"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])
summary.to_excel("./realdata_forest_summary.xlsx")




# real data boosting summarize
log_file_dir = "../realdata_boosting"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))
method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)
summarize_log=pd.DataFrame([])

for method in method_seq:
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    log.columns = "dataset,mse,time,iteration".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)

    
summarize_log = summarize_log[summarize_log["dataset"].isin(report_dataset)]
    
    
summary = pd.pivot_table(summarize_log, index=["dataset"],columns=["method"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])
summary.to_excel("./realdata_boosting_summary.xlsx")





