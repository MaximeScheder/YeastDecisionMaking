# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:16:09 2021

This code is used in order to cluster data into different states

@author: msche
"""

import numpy as np
from autograd.scipy.stats import multivariate_normal as mn


def GMM_EM(centers, sigmas, weights, obs, tolerance=1,
                            Niter = 10, radius = None):
    epsilon = 10
    L_new = 1e10
    iteration = 0
    
    N =obs.shape[0]
    dim = obs.shape[1]
    k = len(weights)
    q = np.zeros((N, k)) #assignment
    
    original_centers =  centers.copy()  
    
    while epsilon > tolerance and iteration < Niter:
        
        # Expectaion step
        
        def ComputeQ(x):
            for i in range(k):
                
                def pdf(x):
                    return weights[i]*mn.pdf(x, centers[i], sigmas[i], allow_singular=True)
                            
                q[:,i] = np.apply_along_axis(pdf, 1, x)

            
            Q = q/np.sum(q, axis=1).reshape(N,1)
            norm = np.sum(Q, axis=0)
            
            return Q, q, norm
        
        Q, q, norm = ComputeQ(obs)
        
        # Marginal likelihood step
        L_old = L_new
        L_new = np.sum(np.log(np.sum(q, axis=1)))
        epsilon = np.abs((L_old-L_new)/L_old)
        print("L_old : {:.3f}".format(L_old) + "L_new : {:.2f}".format(L_new))
        print("Epsilon : ", epsilon)
        
        #M step 
        Mus = Q.T @ obs / norm.reshape(-1,1)
        for i in range(len(centers)):
            centers[i] = Mus[i,:]
            
            if radius is not None:
                r = centers[i]-original_centers[i]
                norm_r = np.linalg.norm(r)
                recompute_Q = False
                
                if norm_r > radius:
                    vec_dir = r/norm_r
                    centers[i] = original_centers[i]+radius*vec_dir
                    recompute_Q = True
                    
                if recompute_Q:
                    Q, q, norm = ComputeQ(obs)
            
            sigma_temp = np.zeros((dim,dim))
            for j in range(N):
                sigma_temp += Q[j,i]*(obs[j,:]-centers[i]).reshape(-1,1)@(obs[j,:]-centers[i]).reshape(1,-1)
             
            sigmas[i] = sigma_temp / norm[i]
            
           
            
            weights[i] = norm[i]/N
            
        iteration += 1
    
    return centers, sigmas, weights, Q