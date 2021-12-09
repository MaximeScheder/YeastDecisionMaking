# -*- coding: utf-8 -*-
"""
The script regroup all functions that serve to simulate data
"""

import numpy as np
from landscapes import field_yeast_fate
import clustering

def fitpam_to_landscape(fit_params, AGN, dim_param):
    """
    Adapat the parameters to a form compatible to the landscape
    param:
        fit_params: particle outputs of ABC SMC. Last position is sigma
        AGN : (Nmedia, Nsubstances) matrix for all media used
        dim_param : Dimension of the landscape parameters
    return:
        lands_parm: (Nmedia, dim_param) containing the landscape for each condition
        sigma : The noise term (float)
    """
    
    parameters = fit_params.reshape(-1)
    
    sigma = fit_params[-1]
    lands_parm = parameters[:-1].reshape(dim_param, -1, order="C")
    lands_parm = AGN @ lands_parm.T
    return lands_parm, sigma
                        

def generate_yeasts_fate_landscape(AGN, parameters, X0, F_lscpe, Nstate, Nmeasure,
                                   centers, sigmas, weights, assignment,
                                   dt, Nsteps, dim_param):
    
    """
    Generate the dataset with the given parameters
    param:
        AGN: (Nconditions, Nvaraiable+1) array for the different media
        parameters : (dim_param * (Nvariable+1)) array for the landscape parameters
        X0 : (Nconditions, Ncells, 2) array of inital conditions
        F_lscpe : field function of the landscape
        Nstate : Number of different states (GMM component)
        Nmeasure : Number of samples
        centers : (Nconditions, Nstate, 2) array of inital centers of GMM
        sigmas : (Nconditions, Nstate, 2, 2) array of covariance matrix
        weights : (Nconditions, Nstate) array of GMM weights
        assignement : (Nconditions, Ncells) array containing the assignement to clusters
        dt : time step in hour
        Nsteps : number of steps for each simulation
        dim_param : number of landscape parameters to fit (only geometrical, excluding noise term for example)
        
    return:
        R : (Nconditions, Nstate, Nmeasure) matrix of proportion of cells in different states
    """
    
    Ncells = X0.shape[0]
    Nconditions = AGN.shape[0]
    R = np.zeros((Nconditions, Nstate, Nmeasure)) 
    
    # Update landscape param according to the condition W0 + A*W1 + ...
    landscape_params, sigma = fitpam_to_landscape(parameters, AGN, dim_param)
    
    
    Xinit = X0

    for k in range(Nmeasure):
        # evolve the cells
        X = euler(Xinit, sigma, F_lscpe, landscape_params, dt, Nsteps, Nconditions)[:,:,:,-1].squeeze()
        Xinit = X
        #cluster cells
        for c in range(Nconditions):
            centers[c], sigmas[c], weights[c], assignment[c] = (
            clustering.GMM_EM(centers[c], sigmas[c], weights[c], X[c],
                              radius=0.2))
        
        for j in range(Nstate):
            #assign the ratio of cells in response matrix
            for c in range(Nconditions):
                R[c,j,k] = np.size(np.where(assignment[c] == j))/Ncells
    
    return R

# This function works for multiple conditions
def euler(X0, sigma, F, parameters, dt, N, Nconditions):
    """
    Do euler Step for stochastic simulation
    param:
        X0 : (Nconditions, Ncells, 2) array of inital values
        sigma : noise parameter (float)
        F : Field function of the form F(x, y, param)
        dt : time step size
        N : Number of steps to do
        Nconditions : number of different media uesed
        
    return:
        Xem: (Nconditions, Ncells,2,N) array of time evolution
    """
    
    Ncells = X0.shape[0]
    dim_param = parameters.shape[1]
    dW = np.sqrt(dt)*np.random.normal(0, 1, (Nconditions, Ncells,2,N))
    Xem = np.zeros((Nconditions, Ncells, 2, N))
    Xtemp = np.zeros((Nconditions, Ncells, 2))
    Xtemp[:,:,0] = X0[:,:,0]
    Xtemp[:,:,1] = X0[:,:,1]
    Xtemp = Xtemp.reshape(Ncells*Nconditions, 2) # put all condition queue to queue
    p = np.ones(Ncells, Nconditions, dim_param)
    p = p*parameters
    p = p.reshape(Ncells*Nconditions, dim_param, order="F") # creat a matrix with same number of lines as Xtemp with corresponding conditions
    
    for i in range(N):
        Field = F(Xtemp[:,:,0], Xtemp[:,:,1], parameters).squeeze().transpose()
        Xtemp = Xtemp + dt * Field + sigma * dW[:,:,i]
        Xem[:,:,:,i] = Xtemp.reshape(Nconditions, Ncells, 2)
    
    return Xem


# ----------------------------------------------------------------------------
# ---------------------------Definition of Prioirs--------------------------
# ----------------------------------------------------------------------------
   

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