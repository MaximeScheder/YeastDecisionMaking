# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:02:26 2021

This script is used to make the inference of the model using the ABC SMC
algorithm.

@author: msche
"""


#------------------------------LYBRARIES-------------------------------------

import numpy as np
from scipy.stats import multivariate_normal
import time
import os
#------------------------------CLASSES---------------------------------------

class Distribution():
    
    def __init__(self):
        pass
    
    def sample(self):
        """return a sample form the distribution"""
        pass
    
    def evaluate(self, x):
        pass
    
    
class Kernel(Distribution):
    
    def __init__(self):
        super().__init__()

    def sample(self, particle):
        pass
    
    def evaluate(self, particle, given_particle):
        pass

class Multivariate_OLMC(Kernel):
    
    def __init__(self, particles = None, weights = None):
        """
        param:
            parameters: Is a (kxM) array with N particles with dimension k
        """
        super().__init__()
        self.particles = particles
        self.weights = weights
        
        
    def update(self, new_particles, new_weights):
        self.particles = new_particles
        self.weights = new_weights/np.sum(new_weights)
        
    def C(self, particle):
        C = np.zeros((particle.size, particle.size))
        if np.size(self.weights) > 1:
            for i in range(np.size(self.weights)):
                v = (self.particles[:,i]-particle).reshape(-1, 1)
                C += self.weights[i] * v @ v.T
        else:
            v = (self.particles-particle).reshape(-1, 1)
            C += self.weights * v @ v.T
        return C
    
    def sample(self, given_particle, N=1):
        particle = np.random.multivariate_normal(given_particle,
                                                 self.C(given_particle),N)
        return particle
    
    def evaluate(self, particle, given_particle):
            p = multivariate_normal.pdf(particle, given_particle, 
                                        self.C(given_particle)+1e-4*np.diag([1,1]))
            return p
    
        
#------------------------------- FUNCTIONS-----------------------------------
    
def ABC_SMC(epsilons, X, N, prior, Kernel, system, distance,
            mp = None, maxiter = 1, update_kernel = True):
    """
    Function that do the ABC_SMC algorithm in order to sample from the distri-
    bution p_eps=T(theta|X)
        param:
            epsilons : liste of decreasing thresholding values of acceptance 
                of size T
            X : Data collected from the experimental distribution.
            N : number of samples through SMC sampling
            prior: distribution that can be sampled from and evaluated
            Kernel: Transition function from one particle to another
            system: A system that can evolve in order to give theoretical samples X
            distance: function that gives a scalar given two samples X, Y
            update_kernel: Set to False if Kernel does not vary in time
        return:
            theta : parameters from p_eps=T
            
    """
    print("Starting Inference of parameters...")
    T = len(epsilons)
    thetas = np.zeros((np.size(prior.sample()), N, T))
    weights = np.ones((N, T))
    tild_selection = np.zeros((N))
    
    for t, eps in enumerate(epsilons):
        
        if t > 0 and update_kernel:
            mask = np.where(tild_selection == 1)
            if np.size(mask) > 0:
                Kernel.update(thetas[:,mask,t-1].squeeze(),
                              weights[mask, t-1].squeeze())
        
        i = 0
        niter = 0
        while i <= N-1:
            # Sampling new particles
            if niter == maxiter:
                k = 0
                
                for st in os.listdir("./outputs"):
                    if st.find("Particles-Run")==0:
                        k += 1
                
                if mp != None:
                    np.save("./outputs/Particles-Run-batch-{}".format(mp), thetas[:,:,:t-1])
                    np.save("./outputs/Weights-Run-batch-{}".format(mp), weights[:,:t-1])
                else:
                    np.save("./outputs/Particles-Run", thetas)
                    np.save("./outputs/Weights-Run", weights)
                raise Exception("The number of iteration reached the maximum in epoch {}".format(t))
            
            if t==0:
                # sample from the given prior distribution
                theta = prior.sample()
                
            else:
                # draw a particle from the old set with its weights
                index = np.random.multinomial(1,weights[:,t-1])
                theta = thetas[:,np.where(index==1),t-1].squeeze()
                
                # Kernelize the selected particle
                not_in_prior = True
                while not_in_prior:
                    # dumy variable checking that we are in the prior
                    theta_new = Kernel.sample(theta)
                    prior_value = prior.evaluate(theta_new)
                    if prior_value > 0:
                        not_in_prior = False
                        theta = theta_new
            
            # update the parameters in the system
            system.update_param(theta)
            # simulate a data set with the current parameters taking only the 
            # last iteration
            Y = system.generate(N)
            # Compute the distance between the two sets
            d = distance(X,Y)
            
            # Updtate the parameters used to change the kernel
            if t < T-1:
                if d <= epsilons[t+1]:
                    tild_selection[i] = 1
                else:
                    tild_selection[i] = 0
            
            # accept the particle if small enough difference
            if d <= eps:
                thetas[:,i,t] = theta.reshape(-1)
                
                if t > 0:
                # computing the corresponding weight
                    def f(x):
                        return Kernel.evaluate(theta, x)
                    
                    norm = np.sum(weights[:, t-1] * np.apply_along_axis(f, 0, 
                                                                        thetas[:,:,t-1]))
                    weights[i,t] = prior_value/norm
                
                i += 1
                niter = 0
            
            niter += 1
            
        # Normalizing weights
        weights[:,t] = weights[:,t]/np.sum(weights[:,t])
        
    return thetas, weights