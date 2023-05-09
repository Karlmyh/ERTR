import numpy as np


class RegressionFunction(object): 
    def __init__(self,f, dim):
        self.functional = f
        self.dim = dim 
    
        try:
            x=np.random.rand(self.dim)
            y=self.functional(x)
            
        except:
            raise ValueError("f should receive {} dimensional numpy ndarray".format(self.dim))

        assert type(y) in [float, int, np.float64, np.float32, 
                           np.float16, np.int64, np.int32, np.int16, np.int8]
            
        
        
    def apply(self, X):
        
        
        return np.array([self.functional(data) for data in X])
    

        