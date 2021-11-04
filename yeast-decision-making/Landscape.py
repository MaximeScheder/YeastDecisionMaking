# -*- coding: utf-8 -*-
"""
This code show the representation of a geometric map as a class. They can be
Combined in order to achieve a larger map. 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
    return x**4 + y**4 + x**3 - 2*x*y**2 - x**2 + a*x + b*y
    
def binaryFlip_F(a, b, x, y):
    x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
    a = 4*x**3 + 3*x**2 -2*y**2 - 2*x + a
    b = 4*y**3 -4*x*y + b
    return -0.71*np.array([[a+b],
                           [-a+b]])

def plotLandscape(x, y, mapp, normMax = 8):
    potential = mapp.V(x, y)
    vec = mapp.F(x, y)

    vx = vec[0].squeeze()
    vy= vec[1].squeeze()
    color = np.abs(vx)+np.abs(vy)
    mask = np.where(color < normMax)

    fig_2d = plt.figure(figsize =(14, 14))
    ax_2d = plt.axes()
    ax_2d.quiver(x[mask], y[mask], vx[mask], vy[mask], color[mask])
    ax_2d.grid()
    
    mask = np.where(color > normMax )
    potential[mask]=np.nan
    ax_2d.contour(x, y, potential, levels=100, cmap=cm.coolwarm)
    
    return fig_2d, ax_2d

#--------------------------------------------------------------
#------------------------MAPS----------------------------------
#--------------------------------------------------------------

class Map:
    """
    Main father class describing the default parameters of a map
    """
    
    def __init__(self, x0=0, y0=0):
        """
        Initialise the map
            param:
                x0, y0: center of the map
        """
        self.g = np.array([]) #list of parameters
        self.x0, self.y0 = x0, y0
    
    def update(self, parameters, ids):
        """
        Function that can access parameters to update them.
            param:
                parameters: list of the value to change
                ids : list of the index of the given parameter
        """
        for i in range(len(ids)):
            self.g[ids[i]] = parameters[i]
    
    def V(self):
        pass
    
    def F(self):
        pass
    
    def displayParameters(self):
        print("Parameters of the map : \n")
        print("-"*10)
        print(self.p)
        
    def center(self, x, y):
        """
        use the actual center of the map to rescale the given x,y to the map
        """
        return x-self.x0, y-self.y0
    
    def recenter(self, new_x, new_y):
        self.x0 = new_x
        self.y0 = new_y
        
    def name(self):
        return "Map"
    
    def getParam(self):
        return self.g

#----------------------------------------------------------------------------#
#----------------------------- SINGLE MAPS ----------------------------------#
#----------------------------------------------------------------------------#
        
class BinaryChoice(Map):    
    def __init__(self, p1 = 0, p2=0, x0=0, y0=0):
        """
        Binary choice landscape
            param:
                p1, p2: doubles that represent the geomtery propreties
                x0, y0: center of the landscape
        """
        
        super().__init__(x0, y0)
        self.g = np.array([p1, p2])
                
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return x**4 + y**4 + y**3 - 4*x**2*y + y**2 - self.g[0]*x + self.g[1]*y
    
    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[4*x**3 - 8*x*y - self.g[0]],
                         [4*y**3 + 3*y**2 - 4*x**2 + self.g[1]]])
    
    def name(self):
        return "BChoice"

class BinaryFlip(Map):
    """
    Landscape that serves to have to fates but connected to each others
        param:
            p1, p2 : geometry parameters as double
            x0, y0 : center of the map
            
        NB : The landscape is rotated of 45 degrees to simplify the  fusion
        afterwards.
    """
    
    def __init__(self, p1=0, p2=0, x0=0, y0=0):
        
        super().__init__(x0, y0)
        self.g = np.array([p1, p2])

    def V(self, x, y):
        x, y = self.center(x, y)
        x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
        return x**4 + y**4 + x**3 - 2*x*y**2 - x**2 + self.g[0]*x + self.g[1]*y
    
    def F(self, x, y):
        x, y = self.center(x, y)
        x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
        a = 4*x**3 + 3*x**2 -2*y**2 - 2*x + self.g[0]
        b = 4*y**3 -4*x*y + self.g[1]
        return -0.71*np.array([[a+b],
                         [-a+b]])
    
    def name(self):
        return "BFlip"
     
    
class CuspY(Map):
    """
    Partiularly useful map for two minimas
    param:
        a, b: geometry landscape parameters
        x0, y0: center of map
    """
    
    def __init__(self, b=0, x0=0, y0=0):
        super().__init__(x0, y0)
        self.g = np.array([b])
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return y**4 - y**2 + self.g[0]*y + 4*x**2/2
    
    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[4*x], [4*y**3 - 2*y + self.g[0]]])
    
    def name(self):
        return "CuspY"
    


class CuspX(Map):
    
    def __init__(self, b=0, x0=0, y0=0):
        super().__init__(x0, y0)
        self.g = np.array([b])
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return x**4 - x**2 + self.g[0]*x + 4*y**2/2
    
    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[4*x**3 - 2*x + self.g[0]], [4*y]])   

    def name(self):
        return "CuspX"

#----------------------------------------------------------------------------#
#----------------------------- FUSION MAPS ----------------------------------#
#----------------------------------------------------------------------------#

class Fusion(Map):
    """
    Fuse multiple maps together with
        param:
            mapp: intial mapp of reference from the class "Map"
            glueFunction: function that connect this map to the rest of the
                landscape. It has to be f(x,y) and compatible with 2D matrices
            velocity: A parameter that represent the scaling factor of the 
                map's associated vectorfield
    """
    
    def __init__(self, mapp, glueFunction, velocity = 1):
        # The map has still a global origin that canbe chosen arbitrarily
        super().__init__(mapp.x0, mapp.y0)
        # list of translation factors
        self.centers = [(mapp.x0, mapp.y0)]
        #list of maps
        self.maps = [mapp]
        self.nMaps = 1
        #list of glue functions to "glue" maps together
        self.glueF = []
        self.glueF.append(glueFunction)
        self.velocities = [velocity]
        self.g = mapp.getParam()
                
    def update(self, parameters, ids, mapids):
        for i in range(len(mapids)):
            self.maps[mapids[i]].update(parameters[i], ids[i])
                    
    def addMap(self, mapp, center, glue, velocity):
        self.centers.append(center)
        self.maps.append(mapp)
        self.glueF.append(glue)
        self.nMaps += 1
        self.velocities.append(velocity)
            
    def displayParameters(self):
        phrase = "Parameter" +"\t" + "| Map" + "\t" + "| Value" + "\t" + "| MapID"
        N = len(phrase)
        print(phrase)
        print("-"*N)
        k=1
        for i in range(self.nMaps):
            param = self.maps[i].getParam()
            for j in range(len(param)):
                print("g{}".format(k) + "\t" + "| {}".format(self.maps[i].name()) + "\t" + "| {:.2f}".format(param[j]) + "\t" + "| {}".format(i))
                print("-"*N)
                k += 1
        
        for i in range(len(self.velocities)):
            print("v{}".format(i+1) + "\t" + "| {}".format(self.maps[i].name()) + "\t" + "| {:.2f}".format(self.velocities[i]) + "\t" + "| {}".format(i))
            print("-"*N)
                
    def V(self, x, y):
        
        if self.nMaps > 1:       
            individuals = []
            for i in range(0, self.nMaps):
                xi, yi = self.centers[i]
                g = self.glueF[i](x,y)
                individuals.append(self.velocities[i]*self.maps[i].V(x-xi,y-yi)*g)
                
            return sum(individuals)
        else:
            xi, yi = self.centers[0]
            return self.maps[0].V(x-xi, y-yi)
    
    def F(self, x, y):
        
        if self.nMaps > 1:
            individuals = []            
                
            for i in range(0, self.nMaps):
                xi, yi = self.centers[i]
                g = self.glueF[i](x,y)
                individuals.append(self.velocities[i]*self.maps[i].F(x-xi,y-yi)*g)

            return sum(individuals)
        else:
            xi, yi = self.centers[0]
            return self.maps[0].F(x-xi, y-yi)
        
        
#----------------------------------------------------------------------------#
#----------------------------- FINAL MAP ----------------------------------#
#----------------------------------------------------------------------------#

"""
Fusion is cool but it seems less efficient in term of computation and I would
rather have one landscape with one defined parameter vector. It is less modular,
but in return it is more accessible and efficient I think.
"""

class YeastFate(Map):
    
    def __init__(self, parameters, x0=0, y0=0):
        
        super().__init__(x0, y0)
        self.g = np.array(parameters)
        
        # Utilitary
        
    def update(self, parameters):
        self.g = np.array(parameters)
            
    def displayParameters(self):    
        n = 10               
        print("parameter |" + " value".ljust(n))
        print("-"*(2*n+1))
        for i in range(self.g.size):
            print("p{}".format(i).ljust(n) + "|" + "{}".format(self.g[i]).ljust(n))
            print("-"*(2*n+1))

        
        # Gluing functions
        
    def glueSm(self, x, y):
        return (np.tanh(-10*(y+1.25))+1)/2
            
    def glueSM(self, x,y):
        return (np.tanh(10*(x-2.65))+1)/2
        
    def glueG1(self, x,y):
        return (np.tanh(10*(x-1.25))+1)/2*(1-self.glueSM(x,y))
        
    def glueG0(self, x,y):
        return (1-self.glueSm(x,y))*(1-(np.tanh(10*(x-1.25))+1)/2)
        
        # Total landscape
        
    def F(self, x, y):
        x,y = self.center(x,y)
        return (self.glueG0(x,y)*self.g[5]*binaryFlip_F(self.g[0],self.g[1],x,y) +
                self.glueSm(x,y)*self.g[6]*cuspY_F(-1,self.g[2], x+0.05, y+1.95) + 
                self.glueG1(x,y)*self.g[7]*cuspX_F(-1,self.g[3], x-1.95, y-0.05) +
                self.glueSM(x,y)*self.g[8]*cuspX_F(-1,self.g[4], x-3.35, y-0.05))
             
    def V(self, x, y):
        x,y = self.center(x,y)
        return (self.glueG0(x,y)*self.g[5]*binaryFlip_V(self.g[0],self.g[1],x,y) +
                self.glueSm(x,y)*self.g[6]*cuspY_V(-1,self.g[2], x+0.05, y+1.95) +
                self.glueG1(x,y)*self.g[7]*cuspX_V(-1,self.g[3], x-1.95, y-0.05) +
                self.glueSM(x,y)*self.g[8]*cuspX_V(-1,self.g[4], x-3.35, y-0.05))
            
      
#----------------------------------------------------------------------------#
#----------------------------- UNUSED MAPS ----------------------------------#
#----------------------------------------------------------------------------#

class DoubleCusp:
    """
    Landscape that is particularly usefull since it has to minima and can switch
    between them only varying one parameter.
    """

    def __init__(self, a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, x0=0, y0=0):
        
        super().__init__(x0, y0)
        self.p = {"dc1":a, "dc2":b, "dc3":c, "dc4":d, "dc5":e, "dc6":f, "dc7":g, "dc8":h}
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return x**4 + y**4 + self.p["dc1"]*x**2*y**2 + self.p["dc2"]*x**2*y + self.p["dc3"]*x*y**2 + self.p["dc4"]*x**2 + self.p["dc5"]*x*y + self.p["dc6"]*y**2 - self.p["dc7"]*x - self.p["dc8"]*y

    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[4*x**3 + 2*self.p["dc1"]*x*y**3 + 2*self.p["dc2"]*x*y + self.p["dc3"]*y**2 + 2*self.p["dc4"]*x + self.p["dc5"]*y - self.p["dc7"]],
                         [4*y**3 + 2*self.p["dc1"]*x**2*y + self.p["dc2"]*x**2 + 2*self.p["dc3"]*x*y + self.p["dc4"]*x + 2*self.p["dc6"]*y - self.p["dc8"]]])
    

class Butterfly(Map):
    
    def __init__(self, a=0, b=0, c=0, d=0, x0=0, y0=0):
        super().__init__(x0, y0)
        self.p = {"b1":a, "b2":b, "b3":c, "b4":d}
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return x**6 + self.p["b1"]*x**4 + self.p["b2"]*x**3 + self.p["b3"]*x**2 + self.p["b4"]*x + y**2/2
    
    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[6*x**5 + 4*self.p["b1"]*x**3 + 3*self.p["b2"]*x**2 + 2*self.p["b3"]*x + self.p["b4"]],
                          [y]])
    
class FoldX(Map):
    
    def __init__(self, a=0, x0=0, y0=0):
        super().__init__(x0, y0)
        self.p = {"fx1":a}
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return x**3 + self.p["fx1"]*x + 4*y**2/2
    
    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[3*x**2 + self.p["fx1"]], [4*y]])
    
class FoldY(Map):
    
    def __init__(self, a=0, x0=0, y0=0):
        super().__init__(x0, y0)
        self.p = {"fy1":a}
        
    def V(self, x, y):
        x, y = self.center(x, y)
        return y**3 + self.p["fy1"]*x + 4*y**2/2
    
    def F(self, x, y):
        x, y = self.center(x, y)
        return -np.array([[4*x]], [3*y**2 + self.p["fy1"]])
