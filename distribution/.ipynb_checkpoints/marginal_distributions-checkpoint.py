#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:17:14 2021

@author: mayuheng
"""



import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import beta
from scipy.stats import laplace
from scipy.stats import expon
from scipy.stats import cosine
from scipy.stats import cauchy
from scipy.stats import t
# super class 

class Distribution(object): 
    def __init__(self):
        pass
    def generate(self, num_samples):
        sample_X = self.sampling(num_samples) 
    
        return sample_X
    def sampling(self, num_samples): 
        pass
    def density(self, sample_X): 
        pass
    
    

    
    
class MultivariateNormalDistribution(Distribution): 
    def __init__(self, mean, cov):
        super(MultivariateNormalDistribution, self).__init__()
        self.mean = mean
        self.cov = cov
        self.dim=np.array([mean]).ravel().shape[0]
        
    def sampling(self, num_samples):
            
        return multivariate_normal.rvs(mean=self.mean,
                                       cov=self.cov,
                                       size=num_samples).reshape(-1,self.dim)
    def density(self, sample_X):
        return multivariate_normal.pdf(sample_X, mean=self.mean, cov=self.cov)

class UniformDistribution(Distribution): 
    def __init__(self, low, upper):
        super(UniformDistribution, self).__init__()
        low=np.array(low).ravel()
        upper=np.array(upper).ravel()
        self.dim=len(low)
        self.low = low
        self.upper = upper
        
    def sampling(self, num_samples):
            
        return np.random.rand(num_samples,self.dim)*(self.upper-self.low)+self.low
    
    def density(self, sample_X):
        in_interval_low=np.array([(sample_X[i]>=self.low).all() for i in range(len(sample_X))])
        in_interval_up=np.array([(sample_X[i]<=self.upper).all() for i in range(len(sample_X))])
        return in_interval_low*in_interval_up/np.prod(self.upper-self.low)


class LaplaceDistribution(Distribution):
    def __init__(self, loc, scale):
        super(LaplaceDistribution, self).__init__()    
        self.loc = loc
        self.scale = scale
        self.dim=np.array(loc).ravel().shape[0]
    def sampling(self, num_samples):
        return np.random.laplace(loc=self.loc, scale=self.scale, size=(num_samples, self.dim))
    def density(self, sample_X):
        return np.prod(laplace.pdf(sample_X, loc=self.loc, scale=self.scale) ,axis=1 )


#class CosineDistribution(Distribution):
#    def __init__(self,scale,loc):
#        super(CosinDistribution, self).__init__()
 #       self.scale = scale
  #      self.loc = loc
   #     self.dim=len(np.array(scale).ravel())
    #def sampling(self, num_samples):
     #   return np.random.beta(a=self.a, b=self.b, size=(num_samples, self.dim))
    #def density(self, sample_X):
     #   return np.prod(beta.pdf(sample_X, a=self.a, b=self.b), axis=1)
  

class BetaDistribution(Distribution):
    def __init__(self, a, b):
        super(BetaDistribution, self).__init__()
        self.a = a
        self.b = b
        self.dim=len(np.array(a).ravel())
    def sampling(self, num_samples):
        return np.random.beta(a=self.a, b=self.b, size=(num_samples, self.dim))
    def density(self, sample_X):
        return np.prod(beta.pdf(sample_X, a=self.a, b=self.b), axis=1)
  
    
class ExponentialDistribution(Distribution):
    def __init__(self, lamda):
        super(ExponentialDistribution, self).__init__()
        
        self.dim=len(np.array(lamda).ravel())
        self.scale=lamda
        
    def sampling(self, num_samples):
        return np.random.exponential(scale=self.scale, size=(num_samples, self.dim))
    def density(self, sample_X):
        return np.prod(expon.pdf(sample_X, scale=self.scale), axis=1)
    
class DeltaDistribution(Distribution):
    def __init__(self, point):
        super(DeltaDistribution, self).__init__()
        
        self.dim=len(np.array(point).ravel())
        self.point=point
        
    def sampling(self, num_samples):
        return np.tile(self.point, num_samples).reshape(num_samples,-1)
    def density(self, sample_X):
        equal_index=np.array([(sample_X[i]==self.point).all() for i in range(len(sample_X))]).ravel()
        result=np.zeros(len(sample_X))
        result[equal_index]=np.inf
        return result
  

class MixedDistribution(Distribution):
    def __init__(self, density_seq, prob_seq):
        super(MixedDistribution, self).__init__()
        self.density_seq = density_seq
        self.prob_seq = prob_seq
        self.num_mix = len(density_seq)
        self.dim=self.density_seq[0].dim
        
    def sampling(self, num_samples):
        rd_idx = np.random.choice(self.num_mix,size=num_samples, 
                                  replace=True, p=self.prob_seq)
        sample_X = []
        for i in range(self.num_mix):
            num_i = np.sum(rd_idx == i)
            density = self.density_seq[i]
            sample_Xi, _ = density.generate(num_i)
            sample_X.append(sample_Xi)
        sample_X = np.concatenate(sample_X, axis=0)
        np.random.shuffle(sample_X)
        return sample_X
    
    # return density at given vector of point
    def density(self, sample_X):
        num_samples = sample_X.shape[0]
        pdf_true = np.zeros(num_samples, dtype=np.float64) 
        for i in range(self.num_mix):
            prob = self.prob_seq[i]
            density = self.density_seq[i]
            pdf_true += prob * density.density(sample_X)
        return pdf_true

    
class CosineDistribution(Distribution):
    def __init__(self,scale,loc):
        super(CosineDistribution, self).__init__()
        self.scale = np.array(scale).ravel()
        self.loc = np.array(loc).ravel()
  
        self.dim=len(np.array(scale).ravel())
    def sampling(self, num_samples):
        
        sample_X=[]
        for i in range(self.dim):
            sample_Xi=cosine.rvs(loc=self.loc[i],scale=self.scale[i],size=num_samples)
            
            
            sample_X.append(sample_Xi.reshape(-1,1))
        sample_X = np.concatenate(sample_X, axis=1)
        return sample_X
    def density(self, sample_X):
        return np.prod(cosine.pdf(sample_X,loc=self.loc,scale=self.scale), axis=1)
 
 
class TDistribution(Distribution):
    def __init__(self,scale,loc,df):
        super(TDistribution, self).__init__()
        self.scale = np.array(scale).ravel()
        self.loc = np.array(loc).ravel()
        self.df=df
  
        self.dim=len(np.array(scale).ravel())
    def sampling(self, num_samples):
        
        sample_X=[]
        for i in range(self.dim):
            sample_Xi=t.rvs(loc=self.loc[i],scale=self.scale[i],size=num_samples,df=self.df)
            
            
            sample_X.append(sample_Xi.reshape(-1,1))
        sample_X = np.concatenate(sample_X, axis=1)
        return sample_X
    def density(self, sample_X):
        return np.prod(t.pdf(sample_X,loc=self.loc,scale=self.scale,df=self.df), axis=1)
 

class CauchyDistribution(Distribution):
    def __init__(self,scale,loc):
        super(CauchyDistribution, self).__init__()
        self.scale = np.array(scale).ravel()
        self.loc = np.array(loc).ravel()
  
        self.dim=len(np.array(scale).ravel())
    def sampling(self, num_samples):
        
        sample_X=[]
        for i in range(self.dim):
         
            sample_Xi=cauchy.rvs(loc=self.loc[i],scale=self.scale[i],size=num_samples)
            
            
            sample_X.append(sample_Xi.reshape(-1,1))
        sample_X = np.concatenate(sample_X, axis=1)
        return sample_X
    def density(self, sample_X):
        return np.prod(cauchy.pdf(sample_X,loc=self.loc,scale=self.scale), axis=1)
 

class UniformCircleDistribution(Distribution):
    def __init__(self, radius):
        super(UniformCircleDistribution, self).__init__()
        
        self.dim=2
        self.radius=radius
        
        
    def sampling(self, num_samples):
        sample=multivariate_normal.rvs(mean=np.zeros(self.dim),
                                       cov=np.diag(np.ones(self.dim)),
                                       size=num_samples).reshape(-1,self.dim)
        for i in range(num_samples):
            sample[i]=sample[i]/np.linalg.norm(sample[i],ord=2)*self.radius
        return sample
    
    def density(self, sample_X):
        pdf=np.array([np.linalg.norm(sample_X[i],ord=2)==self.radius for i in range(len(sample_X))])
        
        return pdf

    
class MarginalDistribution(Distribution):
    def __init__(self, density_object_vector):
        super(MarginalDistribution, self).__init__()   
        self.density_object_vector=density_object_vector
        self.dim_vector=np.array([],dtype='int32')
        for i in range(len(density_object_vector)):
            self.dim_vector=np.append(self.dim_vector,density_object_vector[i].dim)
            
    def sampling(self, num_samples):
        
        sample_X = self.density_object_vector[0].sampling(num_samples)
        for i in range(1,len(self.dim_vector)):
            sample_X=np.hstack([sample_X,self.density_object_vector[i].sampling(num_samples)])
        
        return sample_X        
    def density(self, sample_X):
        pdf_true = np.ones(shape=sample_X.shape[0])
        
        for i in range(len(self.dim_vector)):
            dim_start=np.sum(self.dim_vector[range(i)])
            dim_end=np.sum(self.dim_vector[range(i)])+self.dim_vector[i]
            pdf_true *= self.density_object_vector[i].density(sample_X[:,range(dim_start,dim_end)])
            
        return pdf_true   
