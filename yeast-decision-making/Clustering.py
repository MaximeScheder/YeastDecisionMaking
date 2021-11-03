# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:16:09 2021

This code is used in order to cluster data into different states

@author: msche
"""

import numpy as np
from autograd.scipy.stats import multivariate_normal as mn


def GMM_EM(centers, sigmas, weights, obs, tolerance=0.01,
                            Niter = 10, radius = None):
    """
    This function allows to make a GMM fitting of a 2D data through the Expectation Maximization
    algorithm (EM). It allows to constrain the centers of the clusters into L2 balls
    of a wanted size.
        param:
            centers: list of n the initial centers of the gaussian components
            sigmas: List of the n (2,2) initial covariance matrices 
            weights: n weights in the GMM model associated to each Gaussian
            obs: (N,2) matrix with N observation of 2D positions
            tolerance: wanted precision in the difference between maximum
                likelihood of two steps
            Niter: Maximum number of iterations
            radius: maximum allowed L2 ball radius for the centers to move. If
                None, this constrain is ignored.
        return:
            centers, sigmas, weights : The fitted new lists
            assignment: (N,1) array with values between (0,1,...,n) which
                correspond to the assigned gaussian.
    """
    
    # Initialising parameters
    epsilon = 10
    L_new = 1e10
    iteration = 0
    
    N =obs.shape[0]
    dim = obs.shape[1]
    k = len(weights)
    q = np.zeros((N, k)) #assignment
    
    # making a copy of the old cenetrs 
    original_centers =  centers.copy()  
    
    print("Starting GMM fitting...")
    
    while epsilon > tolerance and iteration < Niter:
        
        # Expectaion step
        
        def ComputeQ(x):
            """
            Function that compute the assignment matrix given the observation
            matrix x.
                return: 
                    Q: the assignment matrix used for further calculations
                    q: unormalized assignment matrix
                    norm: sum of the Lines of Q
            """
            for i in range(k):
                
                # Note that one allows the singular matrices here. By construction
                # The matrices should not be ill-defined and avoid error if 
                # covariance matrix is too small
                def pdf(x):
                    return weights[i]*mn.pdf(x, centers[i], sigmas[i] + 1e-4*np.diag([1, 1]))
                            
                q[:,i] = np.apply_along_axis(pdf, 1, x)

            
            Q = q/np.sum(q, axis=1).reshape(N,1)
            norm = np.sum(Q, axis=0)
            
            return Q, q, norm
        
        Q, q, norm = ComputeQ(obs)
        
        # Marginal likelihood step
        L_old = L_new
        L_new = np.sum(np.log(np.sum(q, axis=1)))
        epsilon = np.abs((L_old-L_new)/L_old) # normalized difference
        
        # Maximization step 
        Mus = Q.T @ obs / norm.reshape(-1,1)
        for i in range(len(centers)):
            centers[i] = Mus[i,:]
            
            if radius is not None:
                # Verify if new center is out of L2 ball
                r = centers[i]-original_centers[i]
                norm_r = np.linalg.norm(r)
                recompute_Q = False
                
                if norm_r > radius:
                    vec_dir = r/norm_r # vector pointing to best direction
                    centers[i] = original_centers[i]+radius*vec_dir # projection on L2 ball
                    recompute_Q = True
                    
                if recompute_Q: #recompute Q only if centers have changed
                    Q, q, norm = ComputeQ(obs)
            
            #Compute the new covariance matrices
            sigma_temp = np.zeros((dim,dim))
            for j in range(N):
                sigma_temp += Q[j,i]*(obs[j,:]-centers[i]).reshape(-1,1)@(obs[j,:]-centers[i]).reshape(1,-1)
             
            sigmas[i] = sigma_temp / norm[i]
            
            # Compute the new weights
            weights[i] = norm[i]/N
            
        iteration += 1
        assignment = np.argmax(Q, axis=1)
        
    print("...GMM fitting is dnone :")
    print("Log-likelihood : {:.2g}".format(L_new))
    print("Tolerance : {:.2g} \n".format(epsilon))
    return centers, sigmas, weights, assignment