import numpy as np
from sklearn.linear_model import LinearRegression

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
    

class PointwiseExtrapolationEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X,
                 dt_Y,
                 order,
                 truncate_ratio_low,
                 truncate_ratio_up,
                 step=1,
                 V = 0,
                 r_range_up=1,
                 r_range_low=0,
                lamda=0.01,
                ):
        self.X_range = X_range

      
        

        self.dim=X_range.shape[1]
        self.dt_X=dt_X
        self.dt_Y=dt_Y
        self.order=order
        self.lamda = lamda
        self.n_node_samples=dt_X.shape[0]
        
        if V == 0:
            if self.n_node_samples > 2000:
                self.truncate_ratio_low = truncate_ratio_low
                self.truncate_ratio_up = max(truncate_ratio_up,
                                            2000/self.n_node_samples + self.truncate_ratio_low)
            else:
                self.truncate_ratio_low=truncate_ratio_low
                self.truncate_ratio_up=truncate_ratio_up
        
        self.dtype = np.float64

        
   
        self.truncate_ratio_low=truncate_ratio_low
        self.truncate_ratio_up=truncate_ratio_up
        self.r_range_up=r_range_up
        self.r_range_low = r_range_low
        self.step = step
        self.V = V
        
        
        
        
    
    def fit(self):

        self.y_hat=None
        
    
        
    def predict(self, test_X,index_by_r=1):
        if index_by_r:
            assert self.V!=0
        
        if len(test_X)==0:
            return np.array([])
        
        
        pre_vec=[]
        for X in test_X:
            if not index_by_r:
                pred_weights,_,_,_,_ = extrapolation_jit(self.dt_X,self.dt_Y, 
                                                  X, self.X_range, self.order,
                                                  self.truncate_ratio_low,self.truncate_ratio_up,
                                                  self.r_range_low,self.r_range_up,self.step,
                                                  self.V,self.lamda)
            else:
                pred_weights,_,_,_,_ = extrapolation_jit_index_r(self.dt_X,self.dt_Y, 
                                                  X, self.X_range, self.order,
                                                  self.truncate_ratio_low,self.truncate_ratio_up,
                                                  self.r_range_low,self.r_range_up,self.step,
                                                  self.V,self.lamda)
            pre_vec.append(pred_weights[0,0])
        y_predict=np.array(pre_vec)
       
        
        return y_predict
    
    def get_info(self, x ,index_by_r=1):
        if index_by_r:
            assert self.V!=0
        
        assert len(x.shape) == 2
        x = x.ravel()
        

        if not index_by_r:
            pred_weights, all_r , all_y_hat ,  used_r, used_y_hat = extrapolation_jit(self.dt_X,self.dt_Y, 
                                              x, self.X_range, self.order,
                                              self.truncate_ratio_low,self.truncate_ratio_up,
                                              self.r_range_low,self.r_range_up,self.step,
                                              self.V,self.lamda)
           
        else:
            
            pred_weights, all_r , all_y_hat ,  used_r, used_y_hat = extrapolation_jit_index_r(self.dt_X,self.dt_Y, 
                                              x, self.X_range, self.order,
                                              self.truncate_ratio_low,self.truncate_ratio_up,
                                              self.r_range_low,self.r_range_up,self.step,
                                              self.V,self.lamda)

        
       
        
        return pred_weights, all_r , all_y_hat  , used_r, used_y_hat, 