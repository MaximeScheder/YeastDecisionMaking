# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:30:37 2021

@author: msche
"""

from landscape import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%% Test Final landscape

mapp = YeastFate([0, 0, 0, 0, 0, 1, 1, 1, 1])

#Ploting and computing vector field
x1, x2 = -1.5, 5
y1, y2 = -4, 1.5
x, y = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 

fig_2d, ax_2d = plotLandscape(x,y, mapp)
ax_2d.annotate("G0", (-0.8, 0.7),size=30)
ax_2d.annotate("G1", (1, 0),size=30)
ax_2d.annotate("SM", (2.5, 0),size=30)
ax_2d.annotate("G0", (3.7, 0),size=30)
ax_2d.annotate("Sm", (-0.3, -1),size=30)
ax_2d.annotate("MI/II", (-0.3, -2.5),size=30)



# Modifying and accessing parameters

mapp.displayParameters()
mapp.update(np.array([1, 0, 0, 1, 0, 1, 1, 1, 1]))
mapp.displayParameters()

#-------------------------------------------------------------------------
#-----------------------------OLD TESTS-----------------------------------
#-------------------------------------------------------------------------
#%% PLot landscaps

mapp = CuspX(-1,0, 3.35, 0.05)
#mapp = Fold(-1)
#mapp = BinaryFlip(-1, 0)
x1, x2 = 2.5, 3
y1, y2 = -0.2, 0.3
x, y = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
xq, yq = np.meshgrid(np.linspace(x1, x2, 20), np.linspace(y1, y2, 20))
potential = mapp.V(x, y)
vec = mapp.F(xq, yq)

fig_2d = plt.figure(figsize =(14, 14))
ax_2d = plt.axes()
ax_2d.quiver(xq, yq, vec[0].squeeze(), vec[1].squeeze())
ax_2d.contour(x, y, potential, levels=100, cmap=cm.coolwarm)

#fig_3d = plt.figure(figsize =(14, 9))
#ax_3d = plt.axes(projection ='3d')
 
# Creating plot
#ax_3d.plot_surface(x, y, potential, cmap=cm.coolwarm)

#%% Test for gluing two landscaps

#First decision landscape with connection from G1 to Sm
map1 = BinaryFlip(0,0)

#State G0 after going through the entire mitotic cycle (furthest right)
map4 = CuspX(0)
c4 = (3.35, 0.05)
def f4(x,y):
    return (np.tanh(10*(x-2.65))+1)/2

#Transition from G1 to SM
map3 = CuspX(0)
c3 = ( 1.95, 0.05)
def f3(x,y):
    return (np.tanh(10*(x-1.25))+1)/2*(1-f4(x,y))


#This landscape is associated to the transition from Sm to MI/II phase at the bottom
map2 = CuspY(0)
c2 = (-0.05, -1.95)
def f2(x,y):
    return (np.tanh(-10*(y+1.25))+1)/2


# Function to glue all the landscape to the initial BinaryFlip 
def f1(x,y):
    return (1-f2(x,y))*(1-(np.tanh(10*(x-1.25))+1)/2)#(1-(np.tanh(10*(y-0.7))+1)/2)#*(1-f2(x,y))


#Adding all functions with their "wights"
d1, d2, d3, d4 = 1, 1, 1, 1
map12 = Fusion(map1, f1, d1)
map12.addMap(map2, c2, f2, d2)
map12.addMap(map3,  c3, f3, d3)
map12.addMap(map4, c4, f4, d4)


# PLoting the landscape 
x1, x2 = -1.5, 5
y1, y2 = -4, 1.5
x, y = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
xq, yq = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
potential = map12.V(x, y)
vec = map12.F(xq, yq)

vx = vec[0].squeeze()
vy= vec[1].squeeze()
color = np.abs(vx)+np.abs(vy)
mask = np.where(color < 8 )

fig_2d = plt.figure(figsize =(14, 14))
ax_2d = plt.axes()
ax_2d.quiver(xq[mask], yq[mask], vx[mask], vy[mask], color[mask])
ax_2d.grid()
ax_2d.annotate("G0", (-0.8, 0.7),size=30)
ax_2d.annotate("G1", (1, 0),size=30)
ax_2d.annotate("SM", (2.5, 0),size=30)
ax_2d.annotate("G0", (3.7, 0),size=30)
ax_2d.annotate("Sm", (-0.3, -1),size=30)
ax_2d.annotate("MI/II", (-0.3, -2.5),size=30)

mask = np.where(color > 8 )
potential[mask]=np.nan
ax_2d.contour(x, y, potential, levels=100, cmap=cm.coolwarm)

#fig_3d = plt.figure(figsize =(14, 9))
#ax_3d = plt.axes(projection ='3d')
 
# Creating plot
#ax_3d.plot_surface(x, y, potential, cmap=cm.coolwarm)
#ax_3d.plot_surface(x, y, 1-np.tanh((x**2+y**2)/3), cmap=cm.coolwarm)


#%% Combined landscape

parameters = np.array([0,0,0,0,0,1,1,1,1])
mapp = YeastFate(parameters)

x1, x2 = -1.5, 5
y1, y2 = -4, 1.5
x, y = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
xq, yq = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
potential = mapp.V(x, y)
vec = mapp.F(xq, yq)

vx = vec[0].squeeze()
vy= vec[1].squeeze()
color = np.abs(vx)+np.abs(vy)
mask = np.where(color < 8 )

fig_2d = plt.figure(figsize =(14, 14))
ax_2d = plt.axes()
ax_2d.quiver(xq[mask], yq[mask], vx[mask], vy[mask], color[mask])
ax_2d.grid()
ax_2d.annotate("G0", (-0.8, 0.7),size=30)
ax_2d.annotate("G1", (1, 0),size=30)
ax_2d.annotate("SM", (2.5, 0),size=30)
ax_2d.annotate("G0", (3.7, 0),size=30)
ax_2d.annotate("Sm", (-0.3, -1),size=30)
ax_2d.annotate("MI/II", (-0.3, -2.5),size=30)

mask = np.where(color > 8 )
potential[mask]=np.nan
ax_2d.contour(x, y, potential, levels=100, cmap=cm.coolwarm)


#%% Animating a landscape

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

N = 30
def animationFunction(frame):
    f = (frame - N/2)/N
    mapp.update({"f1":f})
    ax_2d.cla()
    potential = mapp.V(x, y)
    vec = mapp.F(x, y)
    ax_2d.quiver(x, y, vec[0].squeeze(), vec[1].squeeze())
    ax_2d.contour(x, y, potential, levels=100, cmap=cm.coolwarm)
    ax_2d.set_title("f1 : {}".format(f))
    
anim = FuncAnimation(fig_2d, animationFunction, N)
writervideo = animation.writers["ffmpeg"]
writervideo = writervideo(fps=2)
anim.save('increasingStraightLine.mp4', writer=writervideo)
    
    
