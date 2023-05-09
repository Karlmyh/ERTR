import numpy as np


class GaussianNoise(object): 
    def __init__(self, sigmasq):
        self.sigma = np.sqrt(sigmasq)
        
    def generate(self, n):
        
        return np.random.normal(scale = self.sigma, size =  n)
        

