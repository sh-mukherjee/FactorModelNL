# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:45:41 2019

@author: Norberto
"""

import numpy as np, pandas, AMC
from numba import jit, njit, vectorize, float64

@njit(parallel=True,fastmath=True)
def matrixmult(s,w):
    return s@w

@njit(parallel=True)
def generateRandoms(numAssets,numPortfolios):
    x = np.empty((numPortfolios,numAssets))
    for i in range(numPortfolios):
        for j in range(numAssets):
            x[i,j] = np.random.uniform(0.,1.)
    return x

@vectorize([float64(float64, float64, float64)])
def vectorCrossover(a,b,c):
    return a if c < 0.5 else b

class GeneticAlgorithm:
    
    def __init__(self,sim,maxportfolios,maxgens,maxvol):
        self._sim = sim
        self._numAssets = sim.shape[1]
        self._maxportfolios = maxportfolios
        self._maxgens = maxgens
        self._maxvol = maxvol
       
    def portfolioReturns(self,weights):
        return matrixmult(self._sim,weights) - 1.0

    def portfolioMean(self,portfolioReturns):
        return np.mean(portfolioReturns)

    def portfolioVol(self,portfolioReturns):
        return np.std(portfolioReturns,ddof=1)

    def portfolioScore(self,portfolioReturns):
        return self.portfolioMean(portfolioReturns)-10.0 * np.maximum(self.portfolioVol(portfolioReturns)-self._maxvol,0.0)

    def Crossover(self,l1,l2,r):
            return [a if c < 0.5 else b for (a,b,c) in zip(l1,l2,r)]
       
    def Mutation(self,w,r):
            return [a if b > 0.01 else np.maximum(a + np.random.normal(scale=0.01),0.0) for (a,b) in zip(w,r)]
    
    def createNewGeneration(self,parents):
        pweights = [p[1] for p in parents]
        randomsCrossover = list(generateRandoms(self._numAssets,self._maxportfolios))#[np.random.uniform(size = self._numAssets) for j in range(0,self._maxportfolios)] 
        weights = [self.Crossover(pweights[0],pweights[1],r) for r in randomsCrossover]
        randomsMutation = list(generateRandoms(self._numAssets,self._maxportfolios))#[np.random.uniform(size = self._numAssets) for j in range(0,self._maxportfolios)]        
        weights = [self.Mutation(w,r) for (w,r) in zip(weights,randomsMutation)]
        weights = [w/sum(w) for w in weights]
        returns = [self.portfolioReturns(w) for w in weights]
        scores = [(self.portfolioScore(r),w) for (r,w) in zip(returns,weights)]
        scores.sort(key = lambda s : s[0], reverse=True)
        newparents = scores[0:2]
        return newparents
    
    def optimise(self):
        randoms = list(generateRandoms(self._numAssets,self._maxportfolios))#[np.random.uniform(size = self._numAssets) for j in range(0,self._maxportfolios)]
        weights = [np.log(r)/sum(np.log(r)) for r in randoms]
        returns = [self.portfolioReturns(w) for w in weights]
        scores = [(self.portfolioScore(r),w) for (r,w) in zip(returns,weights)]
        scores.sort(key = lambda s : s[0], reverse=True)
        parents = scores[0:2]
        for j in range(self._maxgens):
            parents = self.createNewGeneration(parents)
        return parents

File = 'MarkData.csv'    
def GenerateSim():
    data = pandas.read_csv(File,header=None)
    returns = np.array(data)[:,1]
    cov = np.array(data)[:,2:]
    corr = np.corrcoef(cov)
    calibration = {'MeanReturns':returns,'Vols': np.sqrt(np.diag(cov)),'Correlation':corr}
    sim = AMC.runSimulation(calibration,1.,1.,10000,True)
    simAtStep = np.array([x[:,1] for x in sim])
    return simAtStep

def RunGeneticAlgorithm(sim,maxportfolios,maxgens,maxvol):
    ga = GeneticAlgorithm(sim,maxportfolios,maxgens,maxvol)
    p = ga.optimise()
    return p

#timeit.timeit('RunGeneticAlgorithm(sim,100,100,0.22)',number=1,globals=globals())
