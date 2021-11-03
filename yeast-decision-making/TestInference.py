# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 12:58:18 2021

@author: msche
"""
from Inference import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from System import *

#%% Testing on a usual case (without simulation)

# Creating the prior for the toy model which is just 
class prior_toy(Distribution):
    
    def __init__(self):
        self.a = -50
        self.b = 50
        
    def sample(self, N=1):
        return np.random.uniform(self.a, self.b, (2, N))
    
    def evaluate(self, particle):
        return np.prod(uniform.pdf(particle, self.a, self.b-self.a))
    

prior = prior_toy()
param1, param2 = 8, 4
N = 50

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

system = toy_system(param1, param2)
X = system.generate(N) # generated data

# Define the kernel used
kernel = Multivariate_OLMC()

# Define the distance used to compare data
def distance(X, Y):
    hx = np.histogram(X, bins=100, range = (-50, 50))[0]
    hy = np.histogram(Y, bins=100, range = (-50, 50))[0]
    return np.sum(np.abs(hx-hy))

epsilons = [160, 120, 80, 60, 40, 30, 20, 15, 10, 8, 7, 6, 4, 3, 2, 1]
#%% Simulating

particles, weights = ABC_SMC(epsilons, X, N, prior, kernel, system, distance,
                             maxiter=1e5)

plt.hist(particles[0,:,-1])
