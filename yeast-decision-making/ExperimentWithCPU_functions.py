# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:35:08 2021

@author: msche
"""
import multiprocessing as mp
from Inference import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from System import *

class prior_toy(Distribution):
    
    def __init__(self):
        self.a = -50
        self.b = 50
        
    def sample(self, N=1):
        return np.random.uniform(self.a, self.b, (2, N))
    
    def evaluate(self, particle):
        return np.prod(uniform.pdf(particle, self.a, self.b-self.a))
    



# define the system that allows to generate new particles
class toy_system():
    
    def __init__(self, a, b):
        self.a, self.b = a, b
    
    def generate(self, N):
        return np.random.normal((self.a-2*self.b)**2 + 
                                             (self.b-4), 1, N)
    
    def update_param(self, param):
        x = param.reshape(-1)
        self.a, self.b = x[0], x[1]
        
def distance(X, Y):
    hx = np.histogram(X, bins=100, range = (-50, 50))[0]
    hy = np.histogram(Y, bins=100, range = (-50, 50))[0]
    return np.sum(np.abs(hx-hy))

def f(X):
    
    prior = prior_toy()
    param1, param2 = 8, 4
    batchsize = 10
    system = toy_system(param1, param2)
    
    
    # Define the kernel used
    kernel = Multivariate_OLMC()
    
    
    epsilons = [160, 120, 80, 60, 40, 30, 20, 15, 10, 8, 7, 6, 4, 3, 2, 1]

    
    return ABC_SMC(epsilons, X, batchsize, prior, kernel, system, distance)