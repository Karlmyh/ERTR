from .marginal_distributions import (LaplaceDistribution, 
                          BetaDistribution,
                          DeltaDistribution,
                          MultivariateNormalDistribution,
                          UniformDistribution,
                          MarginalDistribution,
                          ExponentialDistribution,
                          MixedDistribution,
                          UniformCircleDistribution,
                          CauchyDistribution,
                          CosineDistribution,
                          TDistribution
                          )

from .regression_function import RegressionFunction

from .noise_distributions import GaussianNoise

from .joint_distribution import JointDistribution

import numpy as np
import math


def f_1(x):
    return np.sin(16*x[0])

def f_2(x):
    return np.sin(np.pi*x[0]*x[1]) + 20* (x[2]-0.5)**2 + 10*x[3] + 5 *x[4]

def f_3(x):
    return np.sqrt(x[0]**2 + (x[1]*x[2]-1/x[1]/x[3])**2)

def f_4(x):
    return np.arctan(1/x[0]/(x[1]*x[2]-1/x[1]/x[3]))

def f_5(x):
    return np.sum(x**2)

def f_6(x):
    return x[0]

def f_7(x):
    return np.abs(np.sin(np.pi*2*x[0]))*np.abs(np.sin(np.pi*2*x[1]))*np.abs(np.sin(np.pi*2*x[2]))*np.abs(np.sin(np.pi*2*x[3]))*np.abs(np.sin(np.pi*2*x[4]))

def f_8(x):
    return np.abs(np.sin(np.pi*x[0]))*np.abs(np.sin(np.pi*x[1]))

def f_9(x):
    if int(x[0]*4) % 2 == 0:
        if int(x[1]*4) % 2 == 0:
            return -3
        else:
            return -10
    else:
        if int(x[1]*4) % 2 == 0:
            return 8
        else:
            return 2
        
        
def f_10(x):
    return np.sin(np.pi*x[0]*x[1]) + 20* (x[2]-0.5)**2 + 10*x[3] + 5 *x[4]
        


class TestDistribution(object):
    def __init__(self,index,dim="auto"):
        self.dim=dim
        self.index=index
        
    def testDistribution_1(self):
        if self.dim == "auto":
            self.dim = 1
        assert self.dim == 1
        marginal_obj = UniformDistribution(0,1)
        regression_obj = RegressionFunction(f_1, self.dim)
        noise_obj = GaussianNoise(1)
        
        
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def testDistribution_2(self):
        if self.dim == "auto":
            self.dim = 10
        assert self.dim == 10
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_2, self.dim)
        noise_obj = GaussianNoise(1)
        
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def testDistribution_3(self):
        if self.dim == "auto":
            self.dim = 4
        assert self.dim == 4
        marginal_obj = UniformDistribution(np.array([0,40*np.pi,0,1]), np.array([100,560*np.pi,1,11]))
        regression_obj = RegressionFunction(f_3, self.dim)
        noise_obj = GaussianNoise(1)
        
        X_range = np.array([np.array([0,40*np.pi,0,1]), np.array([100,560*np.pi,1,11])])
        
        return JointDistribution(marginal_obj, regression_obj, noise_obj, X_range)
    
    def testDistribution_4(self):
        if self.dim == "auto":
            self.dim = 4
        assert self.dim == 4
        marginal_obj = UniformDistribution(np.array([0,40*np.pi,0,1]), np.array([100,560*np.pi,1,11]))
        regression_obj = RegressionFunction(f_4, self.dim)
        noise_obj = GaussianNoise(1)
        
        X_range = np.array([np.array([0,40*np.pi,0,1]), np.array([100,560*np.pi,1,11])])
        
        return JointDistribution(marginal_obj, regression_obj, noise_obj, X_range)
    
    def testDistribution_5(self):
        if self.dim == "auto":
            self.dim = 4
        
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_5, self.dim)
        noise_obj = GaussianNoise(1)
        
     
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def testDistribution_6(self):
        if self.dim == "auto":
            self.dim = 2
        
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_6, self.dim)
        noise_obj = GaussianNoise(1)
        
     
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def testDistribution_7(self):
        if self.dim == "auto":
            self.dim = 5
        assert self.dim == 5
        
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_7, self.dim)
        noise_obj = GaussianNoise(0.1)
        
     
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
   

    def testDistribution_8(self):
        if self.dim == "auto":
            self.dim = 2
        assert self.dim == 2
        
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_8, self.dim)
        noise_obj = GaussianNoise(0.1)
        
     
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def testDistribution_9(self):
        if self.dim == "auto":
            self.dim = 2
        assert self.dim == 2
        
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_9, self.dim)
        noise_obj = GaussianNoise(0.1)
        
     
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
   
    def testDistribution_10(self):
        if self.dim == "auto":
            self.dim = 5
        assert self.dim == 5
        marginal_obj = UniformDistribution(np.zeros(self.dim),np.ones(self.dim))
        regression_obj = RegressionFunction(f_2, self.dim)
        noise_obj = GaussianNoise(1)
        
        return JointDistribution(marginal_obj, regression_obj, noise_obj)
    
    def returnDistribution(self):
        switch = {'1': self.testDistribution_1,                
                  '2': self.testDistribution_2,   
                  '3': self.testDistribution_3,   
                  '4': self.testDistribution_4, 
                  '5': self.testDistribution_5, 
                  '6': self.testDistribution_6, 
                  '7': self.testDistribution_7,
                  '8': self.testDistribution_8,
                  '9': self.testDistribution_9,
                  '10': self.testDistribution_10,
          }

        choice = str(self.index)  
        #print(switch.get(choice))                # 获取选择
        result=switch.get(choice)()
        return result
    
