# -*- coding: utf-8 -*-
"""
Here are present the landscape for the euler simulations.
"""

#--------- LYBRARIES

import numpy as np
import matplotlib.pyplot as plt

#--------- LANDSCAPES

def cuspX_V(a, b, x, y):
    return x**4 + a*x**2 + b*x + 4*y**2/2
    
def cuspX_F(a, b, x, y):
    return -np.array([[4*x**3 + a*2*x + b], [4*y]])
        
def cuspY_V(a, b, x, y):
    return y**4 + a*y**2 + b*y + 4*x**2/2
    
def cuspY_F(a, b, x, y):
    return -np.array([[4*x], [4*y**3 + a*2*y + b]])
        
def binaryFlip_V(a, b, x, y):
    x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
    return x**4 + y**4 + 0.1*x**3 - 2*x*y**2 - x**2 + a*x + b*y
    
def binaryFlip_F(a, b, x, y):
    x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
    f1 = 4*x**3 + 0.5*3*x**2 -0.5*2*y**2 - 2*x + a
    f2 = 4*y**3 -0.5*2*x*y + b
    return -0.71*np.array([[f1+f2],
                           [-f1+f2]])

def cusp(x, y, parameter):
    p = parameter
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -np.array([[4*x**3 + p[:,0]*2*x + p[:,1]], [4*y]])

def cycle_F(mu, w, b, x, y):
    r = np.linalg.norm(np.array([x, y]), axis= 0)
    cos = x/r
    sin = y/r
    rdot = mu*r  - np.power(r, 3)
    thetadot = (w + b*np.power(r, 2))*r
    return np.array([[rdot*cos + sin*thetadot], [rdot*sin - cos*thetadot]])


def glueCycle(x, y):
    """Gluing function for the cylce in the final landscape"""
    r = np.linalg.norm(np.array([x-2.2, y-0.2]), axis = 0)
    return (np.tanh(-10*(r-1)) + 1)/2

def glueMI(x, y):
    """Glue the MI phase to the rest of the final landscape"""
    return (np.tanh(-10*(y+1.25))+1)/2

def glueG0(x, y):
    """Glue the inital 3 attractor binary landscape to the rest of the mapps"""
    return (1-glueMI(x,y))*(1-glueCycle(x,y))

def field_yeast_fate(x, y, p):
    """ Final landscape, note that p must be the parameters of the landscape 
    p = [bf1, bf2, csp1, cyc1, cyc2, vbf, vcsp]"""
    if p.ndim == 1:
        p = p.reshape(1, -1)
        
    return (glueG0(x,y)*p[:,5]*binaryFlip_F(p[:,0], p[:,1], x, y) +
            glueMI(x,y)*p[:,6]*cuspY_F(-1, p[:,2], x+0.05, y+1.95) +
            glueCycle(x,y)*cycle_F(p[:,3], p[:,4], 0, x-2.2, y-0.2))
    

#--------- UTULITARY
    
def plotLandscape(x, y, mapp, parameters, normMax = 8):
    
    vec = mapp(x, y, parameters)

    vx = vec[0].squeeze()
    vy= vec[1].squeeze()
    color = np.abs(vx)+np.abs(vy)
    mask = np.where(color < normMax)

    fig_2d = plt.figure(figsize =(14, 14))
    ax_2d = plt.axes()
    ax_2d.quiver(x[mask], y[mask], vx[mask], vy[mask], color[mask])
    ax_2d.grid()
    
    
    return fig_2d, ax_2d
    
