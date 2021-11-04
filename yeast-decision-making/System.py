# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:57:40 2021

This regroup all different necessary steps to make a fitting procedure.

@author: msche
"""

import numpy as np

class System():
    """
    This class regroup all the necessary parts to describe a stochastic
    process. It takes a given landscape and makes it evolve, it makes
    classification and Bayesian inferance using other modules.
    """
    
    def __init__(self, mapp, dt, Nsteps, Ncells = 1, sigma=1):
        """
        param:
            mapp: A geometric landscape from a map class
            dt: time step used
            Nsteps: number of time steps used
            Ncells: number of particle to simulate at the same time
            sigma: variance during the wiener process
        """
        self.N = Nsteps
        self.Ncells = Ncells   
        self.dt = dt
        self.T = dt*Nsteps #Total time of simulation
        self.map = mapp
        self.sigma = sigma
        self.timing = 0
    
    def evolve(self, X0):
        """
        This function evolve the system using a simple EM algorithm
            param:
                X0: (Ncellsx2) array that refers to the inital position
            return:
                Xem: (Ncellsx2xN) array with N representing the number of 
                    steps made.
        """
        
        print("Starting Evolution of the System...")
        dW = np.sqrt(self.dt)*np.random.normal(0, 1, (self.Ncells,2,self.N))
        Xem = np.zeros((self.Ncells, 2, self.N))
        Xtemp = np.zeros((self.Ncells, 2))
        Xtemp[:,0] = X0[:,0]
        Xtemp[:,1] = X0[:,1]
        
        for i in range(self.N):
            Field = self.map.F(Xtemp[:,0], Xtemp[:,1]).squeeze().transpose()
            Xtemp = Xtemp + self.dt * Field + self.sigma * dW[:,:,i]
            Xem[:,:,i] = Xtemp
        
        print("... Evolution ended :")
        self.timing += self.T
        print("Time : {} h \n".format(self.timing))
        return Xem
    
    def generate(self, X0):
        return self.evolve(X0)[:,:,-1]
        
            