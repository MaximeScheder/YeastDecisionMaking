# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:31:08 2021

@author: msche
"""
#asdkasldkaslkdaslkd
import landscapes as maps
import inference as inf
import clustering as clst
import simulation as sim
import numpy as np
from inference import prior_toy

############################################################################
################ LANDSCAPES ################################################
############################################################################

def test_landscape_1():
    p = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    
    #Ploting and computing vector field
    x1, x2 = -1.5, 5
    y1, y2 = -4, 1.5
    x, y = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
    
    fig_2d, ax_2d = maps.plotLandscape(x,y, maps.field_yeast_fate, p)
    ax_2d.annotate("G0", (-0.8, 0.7),size=30)
    ax_2d.annotate("G1", (1, 0),size=30)
    ax_2d.annotate("SM", (2.5, 0),size=30)
    ax_2d.annotate("G0", (3.7, 0),size=30)
    ax_2d.annotate("Sm", (-0.3, -1),size=30)
    ax_2d.annotate("MI/II", (-0.3, -2.5),size=30)
    
def test_landscape_2():
    p = np.array([1, -0.3])
    
    #Ploting and computing vector field
    x1, x2 = -1.5, 1.5
    y1, y2 = -1.5, 1.5
    x, y = np.meshgrid(np.linspace(x1, x2, 50), np.linspace(y1, y2, 50)) 
    
    fig_2d, ax_2d = maps.plotLandscape(x,y, maps.cusp, p)

    
    
############################################################################
################ SIMULATIONS ################################################
############################################################################

    
def test_simulation_1():
    dt = 0.01
    N = 2400
    cells = 10
    sigma = 0.3
    p = [-1, -0.3]
    
    x1, x2 = -1.5, 1.5
    y1, y2 = -1.5, 1.5
    n = 50
    x, y = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, 50))
    
    
    X0 = np.ones((cells, 2)) * np.array([[-0.8, 0]])
    
    def F(x, y, p):
       return maps.cuspX_F(p[0], p[1], x, y)
       
    X = sim.euler(X0, sigma, F, p, dt, N).squeeze()
    
    Xmean = np.mean(X, axis=0)
    

    
    fig, ax = maps.plotLandscape(x, y, F, p, 10)
    ax.plot(Xmean[0,:], Xmean[1,:], 'k-')
    

############################################################################
################ CLUSTERING ###############################################
############################################################################

def test_cluster_1():
    
    p = np.array([1, -0.3])
    
    
    dt = 0.01
    steps = 200
    n_step = 5
    cells = 200
    sigma = 0.2
    
    x1, x2 = -1.5, 1.5
    y1, y2 = -1.5, 1.5
    n = 50
    x, y = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, 50))
    
    
    
    X0 = np.ones((cells, 2)) * np.array([[-0.8, 0]])
    
    #centers = [np.array([-0.5, 0]), np.array([0.7, 0]), np.array([0,0])]
    centers = [np.array([-0.8, 0]), np.array([0, 0]), np.array([0.8, 0])]
    n_centers = len(centers)
    sigmas = [np.array([[1, 0], [0, 1]])]*n_centers
    weights = np.ones(n_centers)/n_centers
    colors = ["k", "g", "b"]
        
    for i in range(n_step):
        
        fig, ax = maps.plotLandscape(x, y, maps.cusp, p, 10)
        X = sim.euler(X0, sigma, maps.cusp, p, dt, steps).squeeze()
        X0 = X[:,:,-1]
    
        centers, sigmas, weights, assign = clst.GMM_EM(centers, sigmas, weights, X0, radius=0.2, Niter=30, tolerance=0.01)
        
        for j in range(n_centers):
            x_a, y_a = X0[np.where(assign==j)].T
            ax.plot(x_a, y_a, marker = 'o', color=colors[j], linestyle="", markersize=12)    
            
############################################################################
################ INFERENCE ###############################################
############################################################################

def test_inference_1():
    
    import time
    import multiprocessing as mp
    from scipy.stats import uniform
    
    # Counting number of worker available
    ncpu = mp.cpu_count()
    
    #defining the prior

    # define the system that allows to generate new particles

        
    prior = prior_toy()
    param1, param2 = 8, 4
    batchsize = 50 # number of sample in each batch
    maxiter = 1e6
    
    # Create the system
    system = toy_system(param1, param2)
    

    
    X = system.generate(batchsize*ncpu) # generated data
    batches = np.split(X, ncpu, axis=0) # Split the data
    
    
    # The espilon ladder definition
    epsilons = [0.95, 0.9]#, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15]#, 0.28, 0.26, 0.23, 0.2]#, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15] #160, 120, 80, 60, 40, 30, 20, 15, 10, 8, 7, 6, 4, 3]
        



    T = len(epsilons)   
    print("{} processes with batch size {} :".format(ncpu, batchsize))
    print("\t Total samples : {}".format(ncpu*batchsize))
    
    for t in range(T):
        print("Starting epoch {} with threshold {}".format(t, epsilons[t]))
        processes = []
        start = time.perf_counter()
        
        if T == T-1:
            inf.ABC_SMC_step(epsilons[t], epsilons[t+1], X, X.shape[0], prior,
                              distance, generate, 0, t, maxiter)
        else:
            inf.ABC_SMC_step(epsilons[t], epsilons[t], X, X.shape[0], prior,
                              distance, generate, 0, t, maxiter)
            
        
        for i, batch in enumerate(batches):
            if t != T-1:
                processes.append(mp.Process(target= inf.ABC_SMC_step, args=(epsilons[t], epsilons[t+1],
                                                                            batch, batchsize, prior,
                                                                            distance, generate, i, t,
                                                                            maxiter, )))
            else:
                processes.append(mp.Process(target= inf.ABC_SMC_step, args=(epsilons[t], epsilons[t],
                                                                            batch, batchsize, prior,
                                                                            distance, generate, i, t,
                                                                            maxiter, )))
    
        for p in processes:
          p.start()
    
        for p in processes:
          p.join()
    
        end = time.perf_counter()
        
        print("\t Finished after {}".format(end-start))
    
    particles = np.concatenate([np.load("./outputs/particles-b{}-e{}.npy".format(i, t)) for i in range(ncpu)], axis=0)
    weights = np.concatenate([np.load("./outputs/weights-b{}-e{}.npy".format(i, t)) for i in range(ncpu)])  
        
    np.save("./outputs/particles", particles)
    np.save("./outputs/weights", weights)
    
class toy_system():
    
    def __init__(self, a, b):
        self.a, self.b = a, b
    
    def generate(self, N):
        return np.random.normal((self.a-2*self.b)**2 + 
                                             (self.b-4), 1, (N,1))
    
    def update_param(self, param):
        x = param.reshape(-1)
        self.a, self.b = x[0], x[1]
        
              

def distance(X, Y):
    hx = np.histogram(X, bins=100, range = (-50, 50))[0]
    hy = np.histogram(Y, bins=100, range = (-50, 50))[0]
    return np.sum(np.abs(hx-hy)/np.sum(hx))
    
def generate(N,theta):
    return np.random.normal((theta[0]-2*theta[1])**2 + (theta[1]-4), 1, N)


if __name__ == '__main__':
    #test_landscape_2()
    test_simulation_1()
    #test_cluster_1()
    #test_inference_1()
    #test_inference_1()
        
  
    