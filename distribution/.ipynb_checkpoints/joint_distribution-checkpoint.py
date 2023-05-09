
import numpy as np


class JointDistribution(object): 
    def __init__(self, marginal_obj, regression_obj, noise_obj, X_range = None):
        self.marginal_obj = marginal_obj
        self.regression_obj = regression_obj
        self.noise_obj = noise_obj
        self.X_range = X_range
        
        if self.X_range is None:
            self.X_range = np.array([np.zeros(self.marginal_obj.dim),np.ones(self.marginal_obj.dim)])
        
        
    def generate(self, n):
        
        X = self.marginal_obj.generate(n)
        Y_true = self.regression_obj.apply(X)
        
        
        X = (X- self.X_range[0])/(self.X_range[1] - self.X_range[0])
        
        return X, Y_true+ self.noise_obj.generate(n)
    
    def generate_true(self, n):
        
        X = self.marginal_obj.generate(n)
        Y_true = self.regression_obj.apply(X)
        
        
        X = (X- self.X_range[0])/(self.X_range[1] - self.X_range[0])
        
        
        return X, Y_true
        
        
        
        
    def evaluate(self, X):
        
        X = X*(self.X_range[1] - self.X_range[0])+self.X_range[0]
        return self.regression_obj.apply(X)
        