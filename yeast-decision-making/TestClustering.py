# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:40:34 2021

@author: msche
"""

from Clustering import *
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from System import *
from landscape import *

#%%

# centers = [np.array([0, 0]), np.array([4, 4]), np.array([2, 2])]
# sigmas = [np.array([[0.5, 0], [0, 0.5]]), np.array([[1, 0], [0, 1]]), np.array([[1, -0.9], [-0.9, 1]])]
# weights = [0.3, 0.3, 0.3]

# X1 = multivariate_normal.rvs(centers[0], sigmas[0], 100, random_state=30)
# X2 = multivariate_normal.rvs(centers[1] ,sigmas[1], 100, random_state=18)
# X3 = multivariate_normal.rvs(centers[2] ,sigmas[2], 100, random_state=18)
# X = np.vstack((X1, X2, X3))
# np.random.shuffle(X)

# fig, ax = plt.subplots(1,2)
# ax[0].plot(X1[:,0], X1[:,1], 'bo')
# ax[0].plot(X2[:,0], X2[:,1], 'ro')
# ax[0].plot(X3[:,0], X3[:,1], 'go')
# ax[0].grid(True)

# c = [np.array([0, -1]), np.array([0, 5]), np.array([1, -1.5])]
# S = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
# w = [0.1, 0.4, 0.5]

# c, S, w, Q = GMM_EM(c, S, w, X, radius=None, Niter=30)
# assignment = np.argmax(Q, axis=1)

# X1 = X[np.where(assignment==0)]
# X2 = X[np.where(assignment==1)]
# X3 = X[np.where(assignment==2)]

# ax[1].plot(X1[:,0], X1[:,1], 'ob')
# ax[1].plot(X2[:,0], X2[:,1], 'or')
# ax[1].plot(X3[:,0], X3[:,1], 'og')


#%% Testing clustering on a landscape with sytsem's simulation

mapp = CuspX(-0.3,0)
dt = 0.01
steps = 200
n_step = 5
cells = 300
sigma = 0.4

x1, x2 = -1.5, 1.5
y1, y2 = -1.5, 1.5
n = 50
x, y = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, 50))


syst = System(mapp, dt, steps, cells, sigma)
X0 = np.ones((cells, 2)) * np.array([[-0.8, 0]])

centers = [np.array([-0.5, 0]), np.array([0.7, 0]), np.array([0,0])]
n_centers = len(centers)
sigmas = [np.array([[1, 0], [0, 1]])]*n_centers
weights = np.ones(n_centers)/n_centers
colors = ["k", "g", "b"]


for i in range(n_step):
    
    fig, ax = plotLandscape(x, y, mapp, 10)
    X = syst.evolve(X0).squeeze()
    X0 = X[:,:,-1]

    centers, sigmas, weights, Q = GMM_EM(centers, sigmas, weights, X0, radius=0.2, Niter=30, tolerance=0.01)
    assign = np.argmax(Q, axis=1)
    
    for j in range(n_centers):
        x_a, y_a = X0[np.where(assign==j)].T
        ax.plot(x_a, y_a, marker = 'o', color=colors[j], linestyle="", markersize=12)    
    
    




