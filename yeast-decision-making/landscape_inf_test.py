# -*- coding: utf-8 -*-
"""
This script will do inference on a simple test-landscape, in order to see
if the computer is able to handle it.
"""

import landscapes
import simulation
import clustering
import inference
import numpy as np
import multiprocessing as mp
import time
import numba


def initializer():
    batchsize = 30
    ncpu = mp.cpu_count()
    Ncells = batchsize * ncpu
    dt = 0.01
    Nsteps = 800
    maxiter = 1e6
    dim_param = 2 # without account of sigma
    Nstate = 3 # left / transitionning / right
    Nmeasure = 3
    centers = [np.array([-0.8, 0]), np.array([0, 0]), np.array([0.8, 0])]
    Covs = [np.array([[1, 0], [0, 1]])] * Nstate
    assignment = None
    weights = np.ones(Nstate)/Nstate
    X0 = np.ones((batchsize, 2)) * [-0.8, 0]    
    # The map data
    mapp = landscapes.cusp
    AGN = np.array([1, 1, 1, 1])
    
    return (batchsize, ncpu, Ncells, dt, Nsteps, maxiter, dim_param, Nstate,
            Nmeasure, centers, Covs, assignment, weights, X0, mapp, AGN)
        

def distance(R1, R2):
    return np.sum(np.abs(R1-R2))

def generate(N, parameters):
    batchsize, ncpu, Ncells, dt, Nsteps, maxiter, dim_param, Nstate, Nmeasure, centers, Covs, assignment, weights, X0, mapp, AGN = initializer()
    return simulation.generate_yeasts_fate_landscape(AGN, parameters, X0,
                                              mapp, Nstate, Nmeasure, centers,
                                              Covs, weights,
                                              assignment, dt, Nsteps, dim_param)

def mix_samples(nbatches):
    particles = np.load("./outputs/particles.npy")
    weights = np.load("./outputs/weights.npy")
    distances = np.load("./outputs/distances.npy")
    ids = np.arange(distances.size)
    np.random.shuffle(ids)
    
    particles = particles[ids,:]
    weights = weights[ids]/np.sum(weights) # normalize weights
    distances = distances[ids]
    
    particles = np.split(particles, nbatches, axis=0)
    weights = np.split(weights, nbatches)
    distances = np.split(distances, nbatches)
    
    for i in range(nbatches):
        np.save("./outputs/particles-mix-b{}".format(i), particles[i])
        np.save("./outputs/weights-mix-b{}".format(i), weights[i]/np.sum(weights[i]))
        np.save("./outputs/distances-mix-b{}".format(i), distances[i])
    
    

#numba.cuda.jit(target="cuda")
def fitting():
    # Simulation parameters
    epsilons = [4, 3.1, 2.9, 2.7, 2.5, 2, 1.4, 0.9, 0.6, 0.4, 0.3, 0.23, 0.175]
    begin = 12
    finish =13
    alpha = 0.5
    
    prior = inference.prior_cusp
    batchsize, ncpu, Ncells, dt, Nsteps, maxiter, dim_param, Nstate, Nmeasure, centers, Covs, assignment, weights, X0, mapp, AGN = initializer()
    
    # defining true parameters
    p = np.array([-1, 0, 0, 0, -0.3, 0, 0, 0, 0.3])
    
    # Generating a dataset
    
    # R = simulation.generate_yeasts_fate_landscape(AGN, p, np.ones((Ncells, 2)) * [-0.8, 0],
    #                                               mapp, Nstate, Nmeasure, centers,
    #                                               Covs, weights,
    #                                               assignment, dt, Nsteps, dim_param)
    
    R = np.load("./R_test.npy") 
    
    import os, os.path
    mypath = "./consol"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
            
    mix_samples(ncpu)
    
    
    T = len(epsilons)   
    print("{} processes with batch size {} :".format(ncpu, batchsize))
    print("\t Total samples : {}".format(ncpu*batchsize))
    
    for t in range(begin, finish):
        print("Starting epoch {} with threshold {}".format(t, epsilons[t]))
        processes = []
        start = time.perf_counter()
          
        # if t != T-1:
        #     inference.ABC_SMC_step(epsilons[t], epsilons[t+1], R, init["batchsize"], prior,
        #                           distance, generate, 0, t, init["maxiter"])
        # else:
        #     inference.ABC_SMC_step(epsilons[t], epsilons[t], R, init["batchsize"], prior,
        #                           distance, generate, 0, t, init["maxiter"])
                
        
        for i in range(ncpu):
                processes.append(mp.Process(target= inference.ABC_SMC_step, args=(epsilons[t],
                                                                            R, batchsize, prior,
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
    distances = np.concatenate([np.load("./outputs/distances-b{}-e{}.npy".format(i, t)) for i in range(ncpu)])
    
    print("Proposition of threshold based on distances distribution")
    print("\t With 0.75 quantile : {:.5f}".format(np.quantile(distances, 0.75)))
    print("\t With 0.50 quantile : {:.5f}".format(np.quantile(distances, 0.50)))
    print("\t With 0.25 quantile : {:.5f}".format(np.quantile(distances, 0.25)))
    print("\t With 0.15 quantile : {:.5f}".format(np.quantile(distances, 0.15)))
    
        
    print("Saving final results")
    np.save("./outputs/particles", particles)
    np.save("./outputs/weights", weights)
    np.save("./outputs/distances", distances)


if __name__ == '__main__':
    fitting()