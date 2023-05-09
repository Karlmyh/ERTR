import numpy as np
from RTER import RegressionTree
from sklearn.metrics import mean_squared_error as MSE

class Ensemble(object):
    def __init__(self, 
                 estimator_fun, 
                 estimator_kargs, 
                 ensemble_num,  
                 ):
        self.estimator_fun = estimator_fun
        self.estimator_kargs = estimator_kargs
        self.ensemble_num = ensemble_num
        self.regs = []
        
        
    def fit(self, X, y):
       
        for i in range(self.ensemble_num):
            self.estimator_kargs["random_state"]=i
            self.regs.append(self.estimator_fun(**self.estimator_kargs))
            self.regs[i].fit(X, y)
            
        
    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        for i in range(self.ensemble_num):
            y_hat +=  self.regs[i].predict(X)
        y_hat/= self.ensemble_num
        return y_hat
    
    
        

class RegressionTreeEnsemble(Ensemble):
    def __init__(self,  ensemble_num=20, splitter="maxedge", estimator="naive_estimator",
                 min_samples_split=2, max_depth=None, log_Xrange=True, random_state=None, order=1,
                 truncate_ratio_low=0 , truncate_ratio_up=1,index_by_r=1, 
                 parallel_jobs=0, r_range_low=0,r_range_up=1):
        
        self.splitter = splitter
        self.estimator = estimator
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
    
        self.log_Xrange = log_Xrange
        self.random_state = random_state
    
        self.truncate_ratio_low=truncate_ratio_low
        
        self.truncate_ratio_up=truncate_ratio_up
        self.index_by_r=index_by_r
        
        self.parallel_jobs = parallel_jobs
        self.r_range_up =r_range_up
        self.r_range_low =r_range_low
        
        estimator_fun = RegressionTree
        estimator_kargs = {"splitter":self.splitter, "estimator":self.estimator, "min_samples_split":self.min_samples_split, 
                           "max_depth":self.max_depth,"log_Xrange":self.log_Xrange ,"random_state":self.random_state,
                           "order":self.order,
                           "truncate_ratio_low":self.truncate_ratio_low,"truncate_ratio_up":self.truncate_ratio_up,
                           "index_by_r":self.index_by_r,"parallel_jobs":self.parallel_jobs,
                           "r_range_low":self.r_range_low,"r_range_up":self.r_range_up} 
        
        super(RegressionTreeEnsemble, self).__init__(estimator_fun=estimator_fun,  estimator_kargs =estimator_kargs, ensemble_num = ensemble_num )
        
        
        
        
        
        
        
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
        for key in [ "ensemble_num" ,'min_samples_split',"max_depth","order", 
                    "truncate_ratio_low","truncate_ratio_up","splitter",
                    "r_range_low","r_range_up"]:
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