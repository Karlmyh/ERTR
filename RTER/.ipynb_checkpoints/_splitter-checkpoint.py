import numpy as np

class PurelyRandomSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range):
        n_node_samples, dim = X.shape
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = np.random.uniform(rddim_min, rddim_max)
        return rd_dim, rd_split
    
class MidPointRandomSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range):
        n_node_samples, dim = X.shape
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split
    
    
class MaxEdgeRandomSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range):
        n_node_samples, dim = X.shape
        edge_ratio= X_range[1]-X_range[0]
        rd_dim = np.random.choice(np.where(edge_ratio==edge_ratio.max())[0])
        #rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split