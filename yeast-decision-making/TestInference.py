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
    

def distance(X, Y):
    hx = np.histogram(X, bins=100, range = (-50, 50))[0]
    hy = np.histogram(Y, bins=100, range = (-50, 50))[0]
    return np.sum(np.abs(hx-hy))

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
        

#%%
if __name__ == '__main__':
    import time
    import multiprocessing as mp
    
    # Counting number of worker available
    ncpu = mp.cpu_count()
    
    #defining the prior
    prior = prior_toy()
    param1, param2 = 8, 4
    batchsize = 10 # number of sample in each batch
    
    # Create the system
    system = toy_system(param1, param2)
    
    X = system.generate(batchsize*ncpu) # generated data
    batches = np.split(X, ncpu, axis=0) # Split the data
    
    # define the kernel function
    kernel = Multivariate_OLMC()
    
    # The espilon ladder definition
    epsilons = [160, 120, 80, 60]#, 40, 30, 20, 15, 10, 8]#, 7, 6, 4, 3]
    
    

    # List to catch batches that stopped earlier 
    exception = [False]*ncpu
    results = []
    
    # Compute task with each worker
    #for i, x in enumerate(batches):
        
        # Raise exception if maximum iteration is catched
        #try:
            
    print("Starting simulation with mulitprocess :")
    start = time.perf_counter()
    
    with mp.Pool(ncpu) as pool:
        for i, x in enumerate(batches):
            
            try:
                results.append(pool.apply(ABC_SMC, (epsilons, x, batchsize, prior ,kernel, 
                                                system, distance, 1, 1e5)))
            except Exception as ex:
                print("\t Batch {}".format(i))
                print(str(ex))
                exception[i] = True
                
    n_ex = np.sum([x==True for x in exception])
    if n_ex > 0 and n_x != ncpu:
        print("ABC_SMC was stuck in some batches, continuing with the remaining {} samples".format((ncpu-n_ex)*batchsize))
        
    # if all have exception we load the last results available
    if all(exception):    
        for i in range(ncpu):
            results.append(np.load("./outputs/Particles-Run-batch-{}.npy".format(i)))
            
    particles = np.concatenate([results[i][0] for i in range(ncpu-n_ex)], axis=1)
    weights = np.concatenate([results[i][1] for i in range(ncpu-n_ex)], axis=0)
    
    np.save("./outputs/Particles-Run-final", particles)
    np.save("./outputs/Weights-Run-final", weights)     
    end = time.perf_counter()
    
        # # if had some exception we ignore those batches
 
    
    print("\t Time for multiprocessing : {:.2f}".format(end-start))
    
    print("Starting regular simulation")
    start = time.perf_counter()
    
    results = ABC_SMC(epsilons, X, batchsize*ncpu, prior ,kernel, 
                                    system, distance, 1, 1e5)
    
    end = time.perf_counter()
    print("\t Time for regular way : {:.2f}".format(end-start))

    

    
    # plt.hist(particles[0,:,-1])
    # plt.hist(particles[1,:,-1])

    
