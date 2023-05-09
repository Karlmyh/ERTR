from numba import njit
import numpy as np

def assign_parallel_jobs(input_tuple):
    idx, node_object, X, numba_acc =input_tuple
    return idx, node_object.predict(X, numba_acc=numba_acc)




@njit
def extrapolation_jit(dt_X, 
                      dt_Y, 
                      X_extra, 
                      X_range, 
                      order, 
                      truncate_ratio_low, 
                      truncate_ratio_up, 
                      r_range_low, 
                      r_range_up, 
                      step, 
                      V,
                      lamda):
    


    radius = (((X_range[1] - X_range[0])**2).sum())**0.5
    n_pts = dt_X.shape[0]
    ratio_vec = np.zeros(n_pts)
    
    for idx_X, X in enumerate(dt_X):
        centralized = X - X_extra
        for d in range(X_extra.shape[0]):
            positive_len = X_range[1,d] - X_extra[d]
            negative_len = X_extra[d] - X_range[0,d]
            
            if centralized[d] > 0:
                centralized[d] /= positive_len
            elif centralized[d] < 0:
                centralized[d] /= negative_len
         
        
        ratio_X = np.abs(centralized).max() 
        ratio_vec[idx_X] = ratio_X
        


    idx_sorted_by_ratio = np.argsort(ratio_vec)  
    
    #### all sorted ratio
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    all_ratio = sorted_ratio.ravel()
    
    #### all sorted y
    sorted_y = dt_Y[idx_sorted_by_ratio]
    pre_vec = np.zeros((n_pts,1))
    for k in range(n_pts):
        pre_vec[k,0] = np.mean(sorted_y[:(k+1)])
    all_y_hat = pre_vec.ravel()
    
    
    #### sort the index by index of samples
    sorted_ratio = sorted_ratio[ int( n_pts * truncate_ratio_low ):int(np.ceil( n_pts * truncate_ratio_up)):step]
    pre_vec = pre_vec[ int( n_pts * truncate_ratio_low ):int(np.ceil( n_pts * truncate_ratio_up)):step]
    
    
    

    
    '''
    index_sorted_by_r = 0
    
    search_r = r_range_low + step_len 
    
    while search_r < 1:
        while index_sorted_by_r < sorted_ratio.shape[0]:
            
            if sorted_ratio[index_sorted_by_r] < search_r:
                index_sorted_by_r += 1
                
            else:
                index_vec_sorted_by_r = np.append(index_vec_sorted_by_r,index_sorted_by_r)
        
     '''
        
    
    
    

    

    
    
    ratio_range_idx_up = sorted_ratio <= r_range_up
    ratio_range_idx_low  = sorted_ratio >= r_range_low
    ratio_range_idx = ratio_range_idx_up * ratio_range_idx_low
    sorted_ratio = sorted_ratio[ratio_range_idx]
    pre_vec = pre_vec[ratio_range_idx]
    
    ratio_mat = np.zeros((sorted_ratio.shape[0], order+1))
    i=0
    while(i < sorted_ratio.shape[0]):
        r= sorted_ratio[i] * radius
        
        for j in range(order +1):
            ratio_mat[i,j] = r**j 
            
        i+=1
        


    
    id_matrix = np.eye( ratio_mat.shape[1] )

    
    ratio_mat_T = np.ascontiguousarray(ratio_mat.T)
    ratio_mat = np.ascontiguousarray(ratio_mat)
    RTR = np.ascontiguousarray(ratio_mat_T @ ratio_mat+ id_matrix * lamda)
    RTR_inv = np.ascontiguousarray(np.linalg.inv(RTR))
    pre_vec = np.ascontiguousarray(pre_vec)
    

    return (RTR_inv @ ratio_mat_T @ pre_vec )[0,0], all_ratio, all_y_hat, sorted_ratio, pre_vec.ravel() 
   
    
    
    

@njit
def extrapolation_jit_index_r(dt_X, 
                      dt_Y, 
                      X_extra, 
                      X_range, 
                      order, 
                      truncate_ratio_low, 
                      truncate_ratio_up, 
                      r_range_low, 
                      r_range_up, 
                      step, 
                      V,
                      lamda):
    


    radius = (((X_range[1] - X_range[0])**2).sum())**0.5
    n_pts = dt_X.shape[0]
    ratio_vec = np.zeros(n_pts)
    
    for idx_X, X in enumerate(dt_X):
        centralized = X - X_extra
        for d in range(X_extra.shape[0]):
            positive_len = X_range[1,d] - X_extra[d]
            negative_len = X_extra[d] - X_range[0,d]
            
            if centralized[d] > 0:
                centralized[d] /= positive_len
            elif centralized[d] < 0:
                centralized[d] /= negative_len
         
        
        ratio_X = np.abs(centralized).max() 
        ratio_vec[idx_X] = ratio_X
        


    idx_sorted_by_ratio = np.argsort(ratio_vec)  
    
    #### all sorted ratio
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    all_ratio = sorted_ratio.ravel()
    
    #### all sorted y
    sorted_y = dt_Y[idx_sorted_by_ratio]
    pre_vec = np.zeros((n_pts,1))
    for k in range(n_pts):
        pre_vec[k,0] = np.mean(sorted_y[:(k+1)])
    all_y_hat = pre_vec.ravel()
    
    
  
    

    #### optimizable
    
    index_by_r = np.zeros(V)
    
    for t in range(V,0,-1):
        if_less_than_r = np.where(sorted_ratio < t/V)[0]
        if len(if_less_than_r) == 0:
            index_by_r[t-1] = index_by_r[t]
        else:
            index_by_r[t-1] = if_less_than_r.max()
            
        
    index_by_r = np.array([int(i) for i in index_by_r])

    pre_vec = pre_vec[index_by_r]
    sorted_ratio = sorted_ratio[index_by_r]
    
    ratio_range_idx_up = sorted_ratio <= r_range_up
    ratio_range_idx_low  = sorted_ratio >= r_range_low
    ratio_range_idx = ratio_range_idx_up * ratio_range_idx_low
    sorted_ratio = sorted_ratio[ratio_range_idx]
    pre_vec = pre_vec[ratio_range_idx]
  
    
    ratio_mat = np.zeros((sorted_ratio.shape[0], order+1))
    i=0
    while(i < sorted_ratio.shape[0]):
        r= sorted_ratio[i] * radius
        
        for j in range(order +1):
            ratio_mat[i,j] = r**j 
            
        i+=1
        


    
    id_matrix = np.eye( ratio_mat.shape[1] )

    
    ratio_mat_T = np.ascontiguousarray(ratio_mat.T)
    ratio_mat = np.ascontiguousarray(ratio_mat)
    RTR = np.ascontiguousarray(ratio_mat_T @ ratio_mat+ id_matrix * lamda)
    RTR_inv = np.ascontiguousarray(np.linalg.inv(RTR))
    pre_vec = np.ascontiguousarray(pre_vec)
    

    return (RTR_inv @ ratio_mat_T @ pre_vec )[0,0], all_ratio, all_y_hat, sorted_ratio, pre_vec.ravel() 
   
    
  
@njit    
def insample_ssq(y):
    return np.var(y)*len(y)
    
    
@njit  
def compute_variace_dim(dt_X_dim, dt_Y):
    
    dt_X_dim_unique = np.unique(dt_X_dim)
    
    if len(dt_X_dim_unique) == 1:
        return np.inf, 0
    

    
    sorted_split_point = np.unique(np.quantile(dt_X_dim_unique,[i/10+0.05 for i in range(10)]))
    
    num_unique_split = len(sorted_split_point)
    
    split_point = 0
    mse = np.inf
    
    for i in range(num_unique_split- 1):
        
        check_split = (sorted_split_point[i]+sorted_split_point[i+1])/2
        

        check_mse = insample_ssq(dt_Y[dt_X_dim<check_split])+insample_ssq(dt_Y[dt_X_dim>=check_split])
    
        if check_mse < mse:
            split_point = check_split
            mse = check_mse
    
    return mse, split_point
