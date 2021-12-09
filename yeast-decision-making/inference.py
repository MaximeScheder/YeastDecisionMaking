# -*- coding: utf-8 -*-
"""
This section regroups all functions that are necessary to perform inference 
with the ABC SMC methode
"""

#------------------
# LYBRARIES
#------------------

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import uniform
from scipy.stats import uniform as unif
from datetime import datetime

#-------------------
# Functions
#-------------------

def multivariate_OLMC(particles, weights, selection):
    """
    This function compute and gives the covariance matrix and the mean for
    the optimal kernel based on last results.
    param:
        particles: all particles of the last iteration of dim (N, d) with 
            N number of sample and d the dimension
        weights: weight associated to the corresponding particles of size
            (N)
        selection: the set of parameters that respect a distance lower than
            the next threshold
    return:
        Cov: (i, j, p) the covariance matrix with i,j = nbr parameters and
            p number of particules
        Means : (1, i, p) the mean associated with the particle p
    """
    
    dim = particles.shape[1]
    N = weights.size
    
    # compute the mean of the particles
    
    index = np.where(selection==1)
    w = weights[index]
    w = w / np.sum(w)
    p_select = particles[index]
    Cov = np.zeros((dim, dim, N))
    
    
    if w.size > 0:
        for i in range(N):
            for k in range(w.size):
                vec = (p_select[k]-particles[i]).reshape(dim, -1)
                Cov[:,:,i] += w[k]*(vec @ vec.T)
                
            Cov[:,:,i] += np.diag(np.ones(dim))*1e-5
    else:
        A = np.zeros((dim, dim))
        for i in range(N):
            for j in range(N):
                vec = (particles[i]-particles[j]).reshape(dim, -1)
                A += weights[i]*weights[j]*(vec @ vec.T)
        
        for i in range(N):
            Cov[:,:,j] = A + np.diag(np.ones(dim))*1e-5
    
    return Cov


def ABC_SMC_step(eps, X, N, prior, 
                 distance, generate, idx=0, epoch=0, maxiter = 1e6):
    """
    This function proceed with one step of the ABC_SMC algorithm
    param:
        eps: current error threshold for distribution
        X : ground truth 
        N : number of particles per batch
        prior : function that either return proba or sample new elements
        distance : function that compute distance between ground truth X and an estimate
        idx : id of the current batch
        epoch : epoch considered
        maxiter : maximum number of iterations
        
        return:
            The algorithm does not return anything but save and load files to/from output folders
        
    """
    w = np.ones(N)/N
    dim = np.size(prior())
    thetas = np.zeros((N, dim))
    distances = np.zeros(N)
    
    if epoch == 0:
        selection = np.zeros(N)

        
    
    if epoch != 0:
        selection = np.zeros(N)
        particles = np.load("./outputs/particles-mix-b{}.npy".format(idx))
        weights = np.load("./outputs/weights-mix-b{}.npy".format(idx))
        d = np.load("./outputs/distances-mix-b{}.npy".format(idx))
        idd = np.where(d < eps)
        selection[idd] = 1
        Cov = multivariate_OLMC(particles, weights, selection)
    else:
        pass
    
    niter = 1
    i = 0
    while i < N:
                
        if niter == maxiter:
            return
        
        if epoch == 0:
            theta = prior()
            
        else:
            # draw a particle from the old set with its weights
            index = np.random.multinomial(1,weights)
            index = np.where(index==1)
            theta = particles[index,:].squeeze()
            
            # Kernelize the selected particle
            not_in_prior = True
            while not_in_prior:
                # dumy variable checking that we are in the prior
                theta_new = np.random.multivariate_normal(theta, Cov[:,:,
                                                                     index].squeeze())
                prior_value = prior(theta_new)
                if prior_value > 0:
                    not_in_prior = False
                    theta = theta_new
                    
            
        # generate the corresponding predictions
        Y = generate(N, theta)
        # compute distance to ground truth
        d = distance(X, Y)
            
        if d <= eps:
            thetas[i,:] = theta.reshape(-1)
            distances[i] = d
            
            if epoch != 0:
            # computing the corresponding weights
                def f(x):
                    return multivariate_normal.pdf(theta, x, Cov[:,:,index].squeeze(), allow_singular=True)
                
                norm = np.sum(weights * np.apply_along_axis(f, 1, particles))
                w[i] = prior_value/norm
            
            
            with open("./consol/follow-b{}-e{}.txt".format(idx, epoch), "a") as file:
                file.write("Sample {}/{} : \n".format(i+1, N))
                file.write("\t niter = {} \n".format(niter))
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                file.write("\t Time : {} \n".format(current_time))
            
            i += 1
            niter = 0
        niter += 1
            
    # Normalizing weights
    w = w/np.sum(w)
    np.save("./outputs/particles-b{}-e{}".format(idx, epoch), thetas)
    np.save("./outputs/weights-b{}-e{}".format(idx, epoch), w)
    np.save("./outputs/distances-b{}-e{}".format(idx, epoch), distances)
        
# ----------------------------------------------------------------------------
# ---------------------------Definition of distances--------------------------
# ----------------------------------------------------------------------------


        
# ----------------------------------------------------------------------------
# ---------------------------Definition of Prioirs--------------------------
# ----------------------------------------------------------------------------
   
def prior_toy(theta=None):
    a = -50
    b = 50
        
    if theta is None:
        return np.random.uniform(a, b, (2, 1))
    
    else:
        return np.prod(uniform.pdf(theta, a, b-a))     
    

def prior_cusp(theta=None):
    a = -2
    b = 2
        
    if theta is None:
        return np.concatenate([np.random.uniform(a, b, (8, 1)),
                               np.random.uniform(0, 1, (1, 1))])
    
    else:
        return np.prod(uniform.pdf(theta[:-1], a, b-a))*(
            unif.pdf(theta[-1], 0, 1))   
        
        
    
    
    