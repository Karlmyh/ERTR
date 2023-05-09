import numpy as np
from sklearn.metrics import mean_squared_error as MSE

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter,MidPointRandomSplitter, MaxEdgeRandomSplitter, VarianceReductionSplitter
from ._estimator import NaiveEstimator,PointwiseExtrapolationEstimator



SPLITTERS = {"purely": PurelyRandomSplitter,"midpoint":MidPointRandomSplitter, "maxedge":MaxEdgeRandomSplitter, "varreduction":VarianceReductionSplitter}
ESTIMATORS = {"naive_estimator": NaiveEstimator,"pointwise_extrapolation_estimator":PointwiseExtrapolationEstimator}

class BaseRecursiveTree(object):
    def __init__(self, 
                 splitter=None, 
                 estimator=None, 
                 min_samples_split=2, 
                 max_depth=None, 
                 order=None, 
                 log_Xrange=None, 
                 random_state=None,
                 truncate_ratio_low=None,
                 truncate_ratio_up=None,
                 index_by_r=None,
                 parallel_jobs=None,
                 step=None,
                 V = None,
                 r_range_up=None,
                r_range_low=None,
                 lamda=None,
                 max_features = None
                ):
        self.splitter = splitter
        self.estimator = estimator
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
        self.step=step
        self.log_Xrange = log_Xrange
        self.random_state = random_state
        self.truncate_ratio_low=truncate_ratio_low
        
        self.truncate_ratio_up=truncate_ratio_up
        self.index_by_r=index_by_r
        
        self.parallel_jobs = parallel_jobs
        self.r_range_up =r_range_up
        self.r_range_low =r_range_low
        self.lamda=lamda
        self.V = V
        
        
        self.max_features = max_features
             
    def fit(self, X, Y,X_range=None):
        self.n_samples, self.n_features = X.shape
        # checking parameters and preparation
        max_depth = (1 if self.max_depth is None
                     else self.max_depth)
        
        if self.V == "auto":
            V =  max(5, int(X.shape[0]* 2**(-max_depth-2)))
        else:
            V = self.V
        
        order= (0 if self.order is None
                else self.order)
        if self.min_samples_split < 1:
            #raise ValueError("min_samples_split should be larger than 1, got {}.".format(self.min_samples_split))
            self.min_samples_split = int(self.n_samples*self.min_samples_split)
        # begin
        splitter = SPLITTERS[self.splitter](self.random_state, self.max_features)
        Estimator = ESTIMATORS[self.estimator]
        self.tree_ = TreeStruct(self.n_samples, self.n_features, self.log_Xrange)
        builder = RecursiveTreeBuilder(splitter, 
                                       Estimator, 
                                       self.min_samples_split, 
                                       max_depth, 
                                       order,
                                       self.truncate_ratio_low,
                                       self.truncate_ratio_up,
                                       self.step,
                                       V,
                                      self.r_range_up,
                                      self.r_range_low,
                                      self.lamda)
        builder.build(self.tree_, X, Y,X_range)
    def apply(self, X):
        return self.tree_.apply(X)
    
    def get_info(self,x):
        return self.tree_.get_info(x)
    
    def predict(self, X):
        if self.parallel_jobs != 0:
            #print("we are using parallel computing!")
            return self.tree_.predict_parallel(X, self.index_by_r,parallel_jobs=self.parallel_jobs)
        else:
            return self.tree_.predict(X, self.index_by_r)


class RegressionTree(BaseRecursiveTree):
    def __init__(self, splitter="maxedge", estimator="pointwise_extrapolation_estimator", min_samples_split=2, 
                 max_depth=None, order=1, log_Xrange=True, random_state=None,truncate_ratio_low=0 , 
                 truncate_ratio_up=1,index_by_r=1,parallel_jobs=0, r_range_low=0,r_range_up=1,step = 1,
                 V = 0,lamda=0.01, max_features = 1.0):
        super(RegressionTree, self).__init__(splitter=splitter, estimator=estimator, 
                                             min_samples_split=min_samples_split,order=order, 
                                             max_depth=max_depth, log_Xrange=log_Xrange, 
                                             random_state=random_state,truncate_ratio_low=truncate_ratio_low,
                                             truncate_ratio_up=truncate_ratio_up,index_by_r=index_by_r,
                                             parallel_jobs=parallel_jobs,r_range_low=r_range_low,
                                             r_range_up=r_range_up,step=step,V=V,lamda=lamda,
                                             max_features=max_features)
    def fit(self, X,Y, X_range="unit"):
        self.dim = X.shape[1]
        if X_range == "unit":
            X_range = np.array([np.zeros(self.dim),np.ones(self.dim)])
        if X_range is None:
            X_range = np.zeros(shape=(2, X.shape[1]))
            X_range[0] = X.min(axis=0)-0.01*(X.max(axis=0)-X.min(axis=0))
            X_range[1] = X.max(axis=0)+0.01*(X.max(axis=0)-X.min(axis=0))
        self.X_range = X_range
        
        super(RegressionTree, self).fit(X,Y,self.X_range)
        
        return self
        
    def predict(self, X):
        
        y_hat = super(RegressionTree, self).predict(X)
        
       
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis=1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis=1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat[np.logical_not(is_inboundary)] = 0
        
        return y_hat
    
    
    
    
    '''
    def get_node_information(self,node_idx,pt_idx):

        querying_object=list(self.tree_.leafnode_fun.values())[node_idx]
        X_extra=querying_object.dt_X[pt_idx]
        sorted_ratio, sorted_prediction, intercept=extrapolation_jit_return_info(querying_object.dt_X,
                                                                                 querying_object.dt_Y,
                                                                                 X_extra, querying_object.X_range,
                                                                                 self.order,self.truncate_ratio_low,
                                                                                 self.truncate_ratio_up)
        return_vec=(querying_object.X_range,
                    querying_object.dt_X,
                    querying_object.dt_Y,
                    sorted_ratio,
                    sorted_prediction,
                    intercept)
        return return_vec
    
    def get_node_extrapolation(self,dt_X, dt_Y, X_extra, X_range, order, low, up,r_low,r_up,step,lamda):
        return extrapolation_jit_return_info(dt_X, dt_Y, X_extra, X_range, order, low, up, r_low,r_up,step,lamda)
    
    '''
    
    def get_node_idx(self,X):
        return self.apply(X)
    
    def get_node(self,X):
        return [self.tree_.leafnode_fun[i] for i in self.get_node_idx(X)]
    
    def get_all_node(self):
        return list(self.tree_.leafnode_fun.values())

    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['min_samples_split',"max_depth","order","truncate_ratio_low",
                    "truncate_ratio_up","splitter","r_range_low","r_range_up",
                    "step","lamda","estimator","V","max_features","index_by_r"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    
    def score(self, X, y):
        
        return -MSE(self.predict(X),y)

