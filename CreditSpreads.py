# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:11:54 2018

@author: norberto
"""

import numpy as np, pandas, matplotlib.pyplot as plt, bisect
from sklearn import decomposition, linear_model
#%matplotlib qt
file = 'C:/Users/Norberto/Documents/ThomsonReutersData/CDS.csv'
spreadmats = np.array([0.5,1,2,3,4,5,7,10])
lgd = 0.75

def spreads(file):
    data = pandas.read_csv(file)
    data.dropna(inplace=True)
    return data

class HullWhiteCalibrationResult:
    def __init__(self,initialMarketCurve,riskNeutralAplha,riskPremiumAlpha,volatility):
        self._initialMarketCurve = initialMarketCurve
        self._riskNeutralAlpha = riskNeutralAplha
        self._riskPremiumAlpha = riskPremiumAlpha
        self._volatility = volatility

class HullWhiteSpreadCalibration:
    def __init__(self,maturities,creditdefaultswapdata,numberofdays):
        self.Maturities = maturities
        self.CreditDefaultSwapData = creditdefaultswapdata
        self.NumberOfDays = numberofdays
        
        self.CalibrationResiduals = np.empty(self.CreditDefaultSwapData.shape[0]-self.NumberOfDays)
        self.InitialMarketCurve = np.empty(self.CreditDefaultSwapData.shape[1])

    def SurvivalProbabilities(self,mats,data):
        #mydata = np.array(data.values,dtype=np.float64)
        x = data
        y = np.diag(mats)
        alldata = np.exp(-np.matmul(x,y))
        self.InitialMarketCurve = alldata[alldata.shape[0]-1,:]
        return alldata
    
    def logReturns(self,bs,days):
        return np.log(np.delete(bs,(0),axis=0)) - np.log(np.delete(bs,(bs.shape[0]-days),axis=0))
    
    def calibrateSpreads(self):
        maturities = self.Maturities
        delta = self.NumberOfDays / 252.0
        data = self.CreditDefaultSwapData
        bs = self.SurvivalProbabilities(maturities,data)
        returns = self.logReturns(bs,self.NumberOfDays)
        shiftedbs = np.log(np.delete(bs,(bs.shape[0]-1),axis=0))
        residuals = np.empty(self.CreditDefaultSwapData.shape[0]-self.NumberOfDays)
        estimated = []
        for j in range (0,bs.shape[1]):
            lm = linear_model.LinearRegression()
            x = lm.fit(shiftedbs[:,j].reshape(-1,1),returns[:,j].reshape(-1,1))
            if maturities[j] == 5.0:
                residuals = returns[:,j].reshape(-1,1)-x.predict(shiftedbs[:,j].reshape(-1,1))
            sigmaB = np.std(returns[:,j].reshape(-1,1)-x.predict(shiftedbs[:,j].reshape(-1,1)),ddof=1)/np.sqrt(delta)
            oneoverBplusalpha = x.coef_[0]/(-1.0 * delta)
            estimated.append(np.array([oneoverBplusalpha, sigmaB],dtype=np.float64))
        
        bvol = list(zip(*estimated))
        self.CalibrationResiduals = residuals
        results = dict([('OneOverBPlusAlpha',bvol[0]),('VolTimesB',bvol[1])])
        rwalpha = np.mean(results['OneOverBPlusAlpha'] - 1.0/spreadmats)
        volalpha = simpleGradientDescent(lambda z: volatilityGradient(z,np.array(results['VolTimesB'])),np.array([0.001,0.001]))
        return HullWhiteCalibrationResult(self.InitialMarketCurve,volalpha[1],rwalpha,volalpha[0])
    #{'OneOverBPlusAlpha':bvol[0],'VolTimesB':bvol[1],'RWAlpha':rwalpha,'Vol':volalpha[0],'RNAlpha':volalpha[1]}
        
        
def bondSpreads(rawspreads,loss):
    rawspreads = rawspreads.drop(['Unnamed: 0'],axis=1)
    mydata = np.array(rawspreads.values,dtype=np.float64)
    x = np.empty((mydata.shape[0],mydata.shape[1]))
    x = mydata/loss/10000
    y = np.diag(spreadmats)
    return np.exp(-np.matmul(x,y))

def bondLogReturns(bs):
    return np.log(np.delete(bs,(0),axis=0)) - np.log(np.delete(bs,(bs.shape[0]-1),axis=0))
    
def volatilityGradient(sa, bv):
    def f(sigma, alpha): return np.array(sigma * (1.0 - np.exp(- alpha * spreadmats)) / alpha)
    x = np.sum((f(sa[0],sa[1]) - bv) * ((1.0 - np.exp(-sa[1] * spreadmats)) / sa[1]))
    y = np.sum((f(sa[0],sa[1]) - bv) * (np.exp(-sa[1] * spreadmats) * (1.0 + sa[1] * spreadmats) - 1.0) * sa[0] / sa[1] / sa[1])
    return 2.0 * np.array([x,y])
    
def simpleGradientDescent(gradientFunc, x0 = np.array([0.0,0.0]), tolerance = 1e-5, iter = 100):
    currentx = x0
    previousStepSize = 1.0
    count = 0
    a = 0.0005
    while(previousStepSize > tolerance and count < iter):
        previousx = currentx
        currentx = currentx - a *  gradientFunc(previousx)
        previousStepSize = np.max(np.abs(currentx - previousx))
        count =  count +1
        
    return currentx
    
def spreadCalibration(spreads, loss=lgd, delta=1.0/252):
    bs = bondSpreads(spreads,loss)
    returns = bondLogReturns(bs)
    shiftedbs = np.log(np.delete(bs,(bs.shape[0]-1),axis=0))
    estimated = []
    for j in range (0,bs.shape[1]):
        lm = linear_model.LinearRegression()
        x = lm.fit(shiftedbs[:,j].reshape(-1,1),returns[:,j].reshape(-1,1))
        sigmaB = np.std(returns[:,j].reshape(-1,1)-x.predict(shiftedbs[:,j].reshape(-1,1)),ddof=1)/np.sqrt(delta)
        oneoverBplusalpha = x.coef_[0]/(-1.0 * delta)
        estimated.append(np.array([oneoverBplusalpha, sigmaB],dtype=np.float64))
    
    bvol = list(zip(*estimated))
#    print(bvol)
    results = dict([('OneOverBPlusAlpha',bvol[0]),('VolTimesB',bvol[1])])
    rwalpha = np.mean(results['OneOverBPlusAlpha'] - 1.0/spreadmats)
#    print(results['VolTimesB'])
    volalpha = simpleGradientDescent(lambda z: volatilityGradient(z,np.array(results['VolTimesB'])),np.array([0.001,0.001]))
    return {'OneOverBPlusAlpha':bvol[0],'VolTimesB':bvol[1],'RWAlpha':rwalpha,'Vol':volalpha[0],'RNAlpha':volalpha[1]}
    
def calcInitialMV(t, marketspreads,loss=lgd):
    rescaledspreads = np.exp(np.log(marketspreads)/loss)
    zeroPlusSpreadMaturities = np.append(0,spreadmats)
    position =  min(bisect.bisect_left(zeroPlusSpreadMaturities,t),len(spreadmats))
    if position == 0:
        a = 1.0
    elif position ==1:
        a = np.exp(t * np.log(rescaledspreads[0])/spreadmats[0])
    else:
        lowerlimit = spreadmats[position - 1]
        distance = t - lowerlimit
        maxposition = min(position,len(spreadmats)-1)
        normalisedDistance = spreadmats[maxposition]-spreadmats[maxposition-1]
        a = rescaledspreads[position-1] * np.exp(distance * np.log(rescaledspreads[maxposition]/rescaledspreads[maxposition-1])/normalisedDistance)
    return a
    
            

plt.plot(fs)
plt.show()

def analyseSpreads(spreads):
    df = np.diff(np.log(spreads),axis=0)
    p = decomposition.PCA(4)
    p.fit_transform(df)
    comp = p.components_
    p.explained_variance_ratio_
    p.explained_variance_ratio_.cumsum()
