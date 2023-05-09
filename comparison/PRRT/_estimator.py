import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

from ._utils import extrapolation_jit, extrapolation_jit_index_r


    

class NaiveEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X, 
                 dt_Y, 
                 order=None,
                 truncate_ratio_low=0,
                 truncate_ratio_up=1,
                 step =1,
                 V = 0,
                 r_range_up=1,
                 r_range_low=0,
                lamda=0.01):
        self.dt_Y=dt_Y
        self.dtype = np.float64
        self.n_node_samples=dt_X.shape[0]
        self.X_range = X_range
        
        
    def fit(self):
        if self.n_node_samples != 0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.y_hat= 0
        
    def predict(self, test_X,index_by_r=0):
        y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        return y_predict
    

    
class PREstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X, 
                 dt_Y, 
                 order=None,
                 truncate_ratio_low=0,
                 truncate_ratio_up=1,
                 step =1,
                 V = 0,
                 r_range_up=1,
                 r_range_low=0,
                lamda=0.01):
        self.dt_Y=dt_Y
        self.dtype = np.float64
        self.n_node_samples=dt_X.shape[0]
        self.X_range = X_range
        self.lamda = lamda
        
        
    def fit(self):
        if self.n_node_samples != 0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.y_hat= 0
        
    def predict(self, test_X):
        y_predict = np.zeros(test_X.shape[0])
        
        for i in range(test_X.shape[0]):
            y_predict[i] = self.y_hat * np.prod([norm.cdf(self.X_range[1,j],loc=test_X[i,j], scale=self.lamda) - norm.cdf(self.X_range[0,j],loc=test_X[i,j], scale=self.lamda) for j in range(test_X.shape[1])])
        return y_predict
    
    
    