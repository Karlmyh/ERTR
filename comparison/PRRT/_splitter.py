import numpy as np
from ._utils import compute_variace_dim

class PurelyRandomSplitter(object):
    def __init__(self, random_state=None,max_features = 1.0 ):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_features = max_features
        
    def __call__(self, X, X_range,dt_Y=None):
        n_node_samples, dim = X.shape
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = np.random.uniform(rddim_min, rddim_max)
        return rd_dim, rd_split
    
class MidPointRandomSplitter(object):
    def __init__(self, random_state=None,max_features = 1.0 ):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_features = max_features
    def __call__(self, X, X_range,dt_Y=None):
        n_node_samples, dim = X.shape
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split
    
    
class MaxEdgeRandomSplitter(object):
    def __init__(self, random_state=None,max_features = 1.0 ):
        self.random_state = random_state
        self.max_features = max_features
        np.random.seed(self.random_state)
    def __call__(self, X, X_range ,dt_Y=None):
        n_node_samples, dim = X.shape
        edge_ratio= X_range[1]-X_range[0]
        
        subsampled_idx = np.random.choice(edge_ratio.shape[0], int(np.ceil(edge_ratio.shape[0]*self.max_features)),replace=False)
        
        rd_dim = np.random.choice(np.where(edge_ratio[subsampled_idx]==edge_ratio[subsampled_idx].max())[0])
        #rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split
    
    
class VarianceReductionSplitter(object):
    def __init__(self, random_state=None,max_features = 1.0 ):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_features = max_features
        
    def __call__(self, X, X_range, dt_Y):
        n_node_samples, dim = X.shape
        subsampled_idx = np.random.choice(dim, int(np.ceil(dim * self.max_features)),replace=False)
        
        
        max_mse = np.inf
        split_dim = None
        split_point = None
        
        for d in range(dim):
            if d in subsampled_idx:
            
                check_mse, check_split_point = compute_variace_dim(X[:,d],dt_Y)
                
                if check_mse < max_mse:
                  
                    max_mse = check_mse
                    split_dim = d
                    split_point = check_split_point
            else:
                continue
                
        if split_point is None:
            print([X,X_range,dt_Y])
        return split_dim, split_point