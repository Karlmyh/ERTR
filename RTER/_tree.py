import numpy as np
from multiprocessing import Pool
from ._utils import assign_parallel_jobs

_TREE_LEAF = -1
_TREE_UNDEFINED = -2

class TreeStruct(object):
    def __init__(self, n_samples, n_features, log_Xrange=True):
        # Base tree statistic
        self.n_samples = n_samples
        self.n_features = n_features
        self.node_count = 0
        # Base inner tree struct
        self.left_child = []
        self.right_child = []
        self.feature = []
        self.threshold = []
        self.n_node_samples = []
        self.leaf_ids = []
        self.leafnode_fun = {}  
        # If log_Xrange is True, the range of each node is also logged.  
        self.log_Xrange = log_Xrange
       
        if log_Xrange == True:
            self.node_range = []
    def _node_append(self):
        self.left_child.append(None)
        self.right_child.append(None)
        self.feature.append(None)
        self.threshold.append(None)
        self.n_node_samples.append(None) 
        if self.log_Xrange == True:
            self.node_range.append(None)  
            
    def _add_node(self, parent, is_left, is_leaf, feature, threshold, n_node_samples, node_range=None):
        self._node_append()
        node_id = self.node_count
        self.n_node_samples[node_id] = n_node_samples
        if self.log_Xrange == True:
            self.node_range[node_id] = node_range.copy()
        # record children status in parent nodes
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.left_child[parent] = node_id
            else:
                self.right_child[parent] = node_id
        # record current node status
        if is_leaf:
            self.left_child[node_id] = _TREE_LEAF
            self.right_child[node_id] = _TREE_LEAF
            self.feature[node_id] = _TREE_UNDEFINED
            self.threshold[node_id] = _TREE_UNDEFINED
            self.leaf_ids.append(node_id)  
        else:
            # left_child and right_child will be set later
            self.feature[node_id] = feature
            self.threshold[node_id] = threshold
        self.node_count += 1
        return node_id
    
    def _node_info_to_ndarray(self):
        self.left_child = np.array(self.left_child, dtype=np.int32)
        self.right_child = np.array(self.right_child, dtype=np.int32)
        self.feature = np.array(self.feature, dtype=np.int32)
        self.threshold = np.array(self.threshold, dtype=np.float64)
        self.n_node_samples = np.array(self.n_node_samples, dtype=np.int32)
        self.leaf_ids = np.array(self.leaf_ids, dtype=np.int32)
        if self.log_Xrange == True:
            self.node_range = np.array(self.node_range, dtype=np.float64)
            
    def apply(self, X):
        return self._apply_dense(X)
    
    def _apply_dense(self, X):
        n = X.shape[0]
        result_nodeid = np.zeros(n, dtype=np.int32)
        for i in range(n):
            node_id = 0
            while self.left_child[node_id] != _TREE_LEAF:
                if X[i, self.feature[node_id]] < self.threshold[node_id]:
                    node_id = self.left_child[node_id]
                else:
                    node_id = self.right_child[node_id]
            result_nodeid[i] = node_id
        return  result_nodeid  
    
    def predict(self, X, index_by_r=1):
        node_affi = self.apply(X)
        y_predict_hat = np.zeros(X.shape[0])
        for leaf_id in self.leaf_ids:
            idx = node_affi == leaf_id
            
            y_predict_hat[idx] = self.leafnode_fun[leaf_id].predict(X[idx], index_by_r=index_by_r)
            
        return y_predict_hat
    
    def get_info(self, x, index_by_r=1):
        
        assert len(x.shape) == 2
        node_affi = self.apply(x)[0]
        
        return self.leafnode_fun[node_affi].get_info(x, index_by_r=index_by_r)
       
    
    def predict_parallel(self, X, index_by_r,parallel_jobs):
        node_affi = self.apply(X)
        y_predict_hat = np.zeros(X.shape[0])
        
        if parallel_jobs == "auto":
            njobs = len(self.leaf_ids)
            #print("using {} threads".format(njobs))
        else:
            njobs = parallel_jobs
        with Pool(njobs) as p:
            result= p.map(assign_parallel_jobs,[ (leaf_id, self.leafnode_fun[leaf_id], X[node_affi == leaf_id],index_by_r) for leaf_id in self.leaf_ids] )

        for return_vec in result:
            idx = node_affi == return_vec[0]
            y_predict_hat[idx]= return_vec[1]
            
        return y_predict_hat
    
    

class RecursiveTreeBuilder(object):
    def __init__(self, splitter, 
                 Estimator, 
                 min_samples_split, 
                 max_depth,order,
                 truncate_ratio_low,
                 truncate_ratio_up,
                 step,
                 V,
                r_range_up,
                r_range_low,
                lamda):
        # about splitter
        self.splitter = splitter
        # about estimator
        self.Estimator = Estimator
        # about recursive splits
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
        self.truncate_ratio_low=truncate_ratio_low
        self.truncate_ratio_up=truncate_ratio_up      
        self.r_range_up=r_range_up
        self.r_range_low = r_range_low
        self.step = step
        self.V = V
        self.lamda = lamda

        
    def build(self, tree, X, Y, X_range=None):
        num_samples = X.shape[0]
        stack = []
        # prepare for stack [X, node_range, parent_status, left_node_status, depth]
        stack.append([X, Y,X_range, _TREE_UNDEFINED, _TREE_UNDEFINED, 0])
        while len(stack) != 0:
            dt_X, dt_Y, node_range, parent, is_left, depth = stack.pop()
            n_node_samples = dt_X.shape[0]
            # judge whether dt should be splitted or not
            if n_node_samples <= self.min_samples_split:
                is_leaf = True
            else:
      
                n_node_unique_samples = np.unique(np.hstack([dt_X,dt_Y.reshape(-1,1)]), axis=0).shape[0]
                
                if depth >= self.max_depth or n_node_unique_samples <= self.min_samples_split:
                    is_leaf = True
                else:
                    rd_dim, rd_split = self.splitter(dt_X, node_range , dt_Y )
                    ## pruning when the sub nodes contains few samples
                    if (dt_X[:,rd_dim] >= rd_split).sum() < self.min_samples_split or (dt_X[:,rd_dim] < rd_split).sum() < self.min_samples_split:
                        is_leaf = True
                    else:
                        is_leaf = False
                    
                
               
                    
                    
            # we will apply splits in non-leaf nodes
            if not is_leaf:
                node_id = tree._add_node(parent, is_left, is_leaf, rd_dim, rd_split, n_node_samples, node_range)
            else:
                node_id = tree._add_node(parent, is_left, is_leaf, None, None, n_node_samples, node_range)
                
                #print([(x<=node_range[1]).all() for x in dt_X])
                tree.leafnode_fun[node_id] = self.Estimator(node_range, 
                                                            num_samples,
                                                            dt_X, 
                                                            dt_Y,
                                                            self.order,
                                                            self.truncate_ratio_low,
                                                            self.truncate_ratio_up,
                                                            self.step,
                                                            self.V,
                                                            self.r_range_up, 
                                                            self.r_range_low,
                                                            self.lamda
                                                           )
        
                tree.leafnode_fun[node_id].fit()
            # begin branching if the node is not leaf
            if not is_leaf:
                # update node range status
                if node_range is not None:
                    node_range_right = node_range.copy()
                    node_range_left = node_range.copy()
                    node_range_right[0, rd_dim] = rd_split
                    node_range_left[1, rd_dim] = rd_split
                else:
                    node_range_right = node_range_left = None
                    
                # push right child on stack
                right_idx = dt_X[:,rd_dim] >= rd_split
                dt_X_right = dt_X[right_idx]
                dt_Y_right = dt_Y[right_idx]
                stack.append([dt_X_right, dt_Y_right , node_range_right, node_id, False, depth+1])
                
                # Push left child on stack
                left_idx = ~right_idx
                dt_X_left = dt_X[left_idx]
                dt_Y_left= dt_Y[left_idx]
                stack.append([dt_X_left, dt_Y_left, node_range_left, node_id, True, depth+1])
                
               
                
        tree._node_info_to_ndarray()
        
        
