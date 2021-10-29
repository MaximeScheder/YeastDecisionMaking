# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:55:42 2021

@author: msche
"""

import numpy as np
import matplotlib.pyplot as plt
from landscape import *
from System import *
import seaborn as sns; sns.set()

#%% Creating a system and making a simulation

# defining the relevant paramters for the simulation
parameters = np.array([-1.5, -1, 0, -2, -0.1, 1, 1, 1, 1])
mapp = YeastFate(parameters)
dt = 0.01
N = 1000
cells = 2
sigma = 0.4

x1, x2 = -1.5, 5
y1, y2 = -4, 1.5
n = 50
x, y = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, 50))

syst = System(mapp, dt, N, cells, sigma)

X0 = np.ones((cells, 2)) * np.array([[-0.8, 0.8]])
X = syst.evolve(X0).squeeze()

Xmean = np.mean(X, axis=0)

fig, ax = plotLandscape(x, y, mapp, 10)
ax.plot(Xmean[0,:], Xmean[1,:], 'k-')

#%%

mapp = CuspX(-0.3,0)
dt = 0.01
N = 1000
cells = 2
sigma = 0.4

x1, x2 = -1.5, 1.5
y1, y2 = -1.5, 1.5
n = 50
x, y = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, 50))

syst = System(mapp, dt, N, cells, sigma)

X0 = np.ones((cells, 2)) * np.array([[-0.8, 0.8]])
X = syst.evolve(X0).squeeze()

Xmean = np.mean(X, axis=0)

fig, ax = plotLandscape(x, y, mapp, 10)
ax.plot(Xmean[0,:], Xmean[1,:], 'k-')
