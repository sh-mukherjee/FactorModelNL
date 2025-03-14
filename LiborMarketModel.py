# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:29:55 2019

@author: Norberto
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import LMM

class LiborMarketModel:    
    def forwardsFromBonds(self,zcb,tenors):
        return (zcb[0:(len(zcb)-1)] / zcb[1:] - 1.0) / tenors
#    def calculateLambdaLambdaT(self,comps,vols):
#            Lambda = np.matmul(comps, np.diag(vols))
#            return np.matmul(Lambda,np.transpose(Lambda))
        
    def __init__(self,initialMaturities,initialZCByieldcurve,principalComponents,Volatilities):
        splitPoints = [12]
        extendedCalibration = LMM.extendCalibration(initialMaturities,initialZCByieldcurve,principalComponents,Volatilities,splitPoints,None)
        self._maturities = extendedCalibration['Maturities'] 
        self._regularmaturities = [m for m in self._maturities if (m%1)==0]
        self._irregularmaturities = [m for m in self._maturities if (m%1)!=0]
        self._initialZCByieldcurve = extendedCalibration['ZCBInitialCurve']
        #self._principalComponents = principalComponents
        self._volatilities = Volatilities
        self._ncomp = principalComponents.shape[1]
        self._time = 0.0
        self._tenors = np.diff(self._maturities)        
        self._regulartenors = [t for t in self._tenors if (t%1)==0]
        self._irregulartenors = [t for t in self._tenors if (t%1)==0]
        #initialVolatilities = principalComponents * self._volatilities
        self._fixedlambda = extendedCalibration['_lambdas']
        self._fixedlambdalambdat = np.matmul(self._fixedlambda,np.transpose(self._fixedlambda))
        self._lambda = extendedCalibration['_lambdas']
        self._lambdalambdat = np.matmul(self._lambda,np.transpose(self._lambda))
        self._forwards = self.forwardsFromBonds(self._initialZCByieldcurve,self._tenors)
        self._martingaleTestResults = [(1.0,1.0,1.0)]
        self._liborHasSplit = False
        self._count = 0
        #self._fixedLibor = (self._tenors[0],self._forwards[0])
        
    def trim2Darray(self,x,count):
        return x[0:(x.shape[0]-count),0:(x.shape[1]-count)]
    
    def bondsFromLibors(self,fixedlibor,libors,tenors):
        maturity = fixedlibor[0] - self._time
        tenors = np.concatenate(([maturity],tenors),0)
        maturities = np.cumsum(tenors)
        libors = np.concatenate(([fixedlibor[1]],libors),0)
        bonds = np.cumprod(1.0 / (1.0 + tenors * libors))
#        if self._time > 1.0:
#        print(fixedlibor)#[0],libors,tenors,maturities
#            print(maturities,bonds)
        if maturity > 0:
            bonds = np.concatenate(([1.0],bonds),0)
            maturities = np.concatenate(([0.0],maturities),0)
        return interp1d(maturities,bonds,kind='cubic') #np.cumprod(1.0 / (1.0 + tenors * libors))
    
    def calculateDrifts(self, rates, tenors, lambdalambdat, width):
        n = len(rates)
        multipliers = tenors / (1.0 + tenors * rates)
        mukj = lambdalambdat * multipliers
        def fz(m,j):
            return np.sum(m[0:(j+1),j])
        individualDrifts =  np.array([fz(mukj,j) for j in range(n)])
#        for j in range(len(rates)):
#            driftsum = 0.0
#            for k in range(j+1):
#                driftsum += lambdalambdat[width * k + j] * multiplier[k]
#            individualDrifts[j] = driftsum
        return individualDrifts
    
    def evolveLibor(self,forwards,drifts,shocks,stepSize=1.0/12.0,sqrtStepSize=1.0/np.sqrt(12.0)):
        arrayToReturn = forwards + drifts * stepSize
        shocks *= sqrtStepSize
        arrayToReturn += np.matmul(self._lambda,shocks)
        return arrayToReturn
        
    def simulateLiborStep(self,forwards,llt, shocks, stepSize=1.0/12.0,sqrtStepSize=1.0/np.sqrt(12.0)):   
        drifts = self.calculateDrifts(forwards,self._tenors,llt,len(forwards))
        #shocksAtStep = shocks * sqrtStepSize
        return self.evolveLibor(forwards,drifts,shocks,stepSize,sqrtStepSize)
       
    def splitLibor(self,libor,tenor,numSplitPoints):#improve this procedure       
        splitforwards = numSplitPoints * (np.exp(np.log(1.0 + libor * tenor) / numSplitPoints) - 1.0) / tenor * np.ones(numSplitPoints - 1)
        #print(libor,tenor,splitbonds)
        return splitforwards
        
    def updateModelVariables(self,forwardlist,fixedliborlist):
        if len(self._irregularmaturities) > 0 and self._time > self._irregularmaturities[0] :
            self._lambda = np.delete(self._lambda,len(self._irregulartenors)-1,0)
            self._lambdalambdat = np.delete(np.delete(self._lambdalambdat,len(self._irregulartenors)-1,0),len(self._irregulartenors)-1,1)
            self._irregulartenors = self._irregulartenors[1:]
            self._irregularmaturities = self._irregularmaturities[1:]    
            self._tenors = self._tenors[1:]
            fixedliborlist = [(self._irregularmaturities[0],f[0]) for f in forwardlist] if len(self._irregularmaturities) > 0 else [(self._time,f[0]) for f in forwardlist]
            forwardlist = [f[1:] for f in forwardlist]
            #print(fixedliborlist[0])
        if self._time > self._regularmaturities[0] :
            self._irregularmaturities = [m+self._regularmaturities[0] for m in self._maturities if (m%1)!=0]
            self._regulartenors = self._regulartenors[1:]
            self._regularmaturities = self._regularmaturities[1:]    
            self._tenors = np.diff(np.concatenate((self._irregularmaturities,self._regularmaturities),0))
            forwardlist = [np.concatenate((self.splitLibor(f[0],self._regulartenors[0],12),f[1:])) for f in forwardlist] 
            self._liborHasSplit = True
            self._count += 1
            fixedliborlist = [(self._irregularmaturities[0],f[0]) for f in forwardlist]
            #print(fixedliborlist[0])
            #splitVols = LMM.splitForwardVols(self._lambda[0:1,:],[12])#Think about the vol split - R example
            self._lambda = self._fixedlambda[0:(self._fixedlambda.shape[0]-self._count),:]#np.concatenate((splitVols,self._lambda[1:(self._lambda.shape[0]-1),:]),0)
            self._lambdalambdat = self.trim2Darray(self._fixedlambdalambdat,self._count)#np.matmul(self._lambda,np.transpose(self._lambda))
        return forwardlist,fixedliborlist        
        
    def runMartingaleTest(self,prevAssetSim,prevNumeraireSim,assetSim,numeraireSim,conflevel):
        ratioBetweenSteps = assetSim  * prevNumeraireSim / numeraireSim / prevAssetSim
        #print(self._time,np.mean(assetSim),np.mean(numeraireSim))
        sampleSize = len (ratioBetweenSteps)
        standardError = np.std(ratioBetweenSteps,ddof=1) / np.sqrt(sampleSize) if sampleSize > 1 else 0.0
        q = (1.0 + conflevel)/2.0
        quantile = norm.ppf(q)
        sampleMean = np.mean(ratioBetweenSteps)
        return (sampleMean - standardError * quantile, sampleMean, sampleMean + standardError * quantile)
        
    def simulateLiborPath(self,steps,trials,martingaleTest):
        runTest =  martingaleTest[0]
        bondMaturity = martingaleTest[1]
        stepsizes = np.diff(steps)
        numSteps = len(stepsizes)
        forwards = self.forwardsFromBonds(self._initialZCByieldcurve,self._tenors)
        fixedLibor = (self._tenors[0],self._forwards[0])
        forwardList = [forwards] * trials 
        fixedLiborList = [fixedLibor] * trials
        #print(forwardList[0][0:2],fixedLiborList[0])
        if runTest:
            previousBankAccount = [1.0] * trials
            previousBondValue = [self._initialZCByieldcurve[np.where(self._maturities == bondMaturity)[0][0]]] * trials
            previousBondValuation = [interp1d(self._maturities,self._initialZCByieldcurve,kind='cubic')] * trials
        #print(previousBankAccount[0],previousBondValue[0],[b(1/12) for b in previousBondValuation][0])
        corr = np.diag(np.ones(self._ncomp))
        shocks = np.random.multivariate_normal(np.zeros(self._ncomp),corr,trials*numSteps)
        #shocks *= self._volatilities 
        shocks.tolist()
        for j in range(numSteps):   
            self._time += stepsizes[j]
            forwardList,fixedLiborList = self.updateModelVariables(forwardList,fixedLiborList)
            #print(forwardList[0][0:2])
            if self._liborHasSplit == True:
                previousBondValuation = [lambda x, f=f: 1.0 / (1.0 + f[0]/12.0) for f in forwardList]
                #print([b(1/12) for b in previousBondValuation],[f for f in forwardList])
            shocksAtStep = shocks[(j*trials):((j+1)*trials)]
            forwardList = [self.simulateLiborStep(f,self._lambdalambdat,s,stepsizes[j],np.sqrt(stepsizes[j])) for f,s in zip(forwardList,shocksAtStep)]
            #print(forwardList[0][0:2])
            self._liborHasSplit = False
            if runTest:
                bondMaturity -= stepsizes[j]
                bondValuation = [self.bondsFromLibors(fl,f,self._tenors) for fl,f in zip(fixedLiborList,forwardList)]
                currentBankAccount = [c/b(1/12) for c,b in zip(previousBankAccount,previousBondValuation)]
                currentBondValue = [b(bondMaturity) for b in bondValuation]
                #print(currentBankAccount[0],currentBondValue[0],[b(1/12) for b in previousBondValuation][0])
                testresult = self.runMartingaleTest(np.array(previousBondValue), np.array(previousBankAccount),np.array(currentBondValue),np.array(currentBankAccount),0.95)
                self._martingaleTestResults.append(testresult)
                previousBankAccount = currentBankAccount
                previousBondValue = currentBondValue     
                previousBondValuation = bondValuation
        #return forwardList
        
def RunLiborMarketModel(steps,trials=10):
    cal = LMM.calibrateModel()
    lmm = LiborMarketModel(np.arange(1,21),cal['ZCBInitialCurve'],cal['Components'],cal['Volatilities'])
    lmm.simulateLiborPath(steps,trials,(True,3))
    testResults = lmm._martingaleTestResults #list(zip(*lmm._martingaleTestResults))
    return testResults
        
    