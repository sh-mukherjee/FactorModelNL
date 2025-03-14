# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np, bisect
import pandas
import matplotlib.pyplot as plt
from sklearn import decomposition, linear_model
#%matplotlib qt

def spotyc():
    data = pandas.read_csv('C:/Users/Norberto/Documents/BankOfEnglandData/BoE_YC.csv')
    data = data.dropna()
    return data

def shortenyc(spotcurve):
    maturities =np.array(range(1,21),dtype=np.str)
    return spotcurve[maturities]

def calculateBondYields(spotcurve,maturities):
    spotyc = np.array((spotcurve.values),dtype=np.float64)
    spotyc /= 100
    bonds = np.empty((spotyc.shape))
    for j in range(0,spotyc.shape[1]):
        bonds[:,j] = (1+spotyc[:,j])**(-maturities[j])
    return bonds
   
def calculateForwards(bonds,maturities):
    tenors = np.diff(maturities)
    forwards = np.empty((bonds.shape[0],tenors.shape[0]))
    for j in range(0,tenors.shape[0]):
        forwards[:,j] = (bonds[:,j]/bonds[:,j+1] - 1) / tenors[j]
    return forwards
        
def calibrateModel(ncomp=4,delta=1/252,maturities=np.array(range(1,21))):
    ycurve = shortenyc(spotyc())
    byc = calculateBondYields(ycurve,maturities)
    forwardcurve = calculateForwards(byc,maturities)
    df = np.diff(forwardcurve,axis=0)
    p = decomposition.PCA(ncomp)
    p.fit_transform(df)
    biglambda = np.transpose(p.components_)
    vols = p.singular_values_ / np.sqrt(df.shape[0]) #np.std(loadings,axis=0)
    projections = np.matmul(forwardcurve,biglambda[:,0:4])
    plesslast = projections[:-1]
    changes = np.diff(projections,axis=0)
    ab = np.empty((3,4))
    for j in range(0,4):
        lm = linear_model.LinearRegression()
        x = lm.fit(plesslast[:,j].reshape(-1,1),changes[:,j].reshape(-1,1))
        a = -x.coef_[0] / delta
        b = x.intercept_[0] / a / delta
        #print(a)
        #print(b)
        ab[0,j] = a
        ab[1,j] = b
        ab[2,j] = a*b
    #a = np.matrix(np.array(a))
    return {'ZCBInitialCurve':byc[byc.shape[0]-1,:],'InitialCurve':forwardcurve[forwardcurve.shape[0]-1,:],'Maturities':maturities,'Components':biglambda,'Volatilities':vols,'A':ab[0,:],'B':ab[1,:],'AB':ab[2,:]}

def splitForwardVols(_lambdas,splitPoints):
    def createPairs(x):
        return [(x[j],x[j+1]) for j in range(len(x)-1)]
    def splitSegment(a,b,n):
        d = 0.5 * (b-a)
        return [b - d + j * (b-a) / (n - 1) for j in range(n)]
    def vectorSplit(v,npoints):
        initialSplit = splitSegment(v[0],1.25 * v[0],splitPoints[0]-1)
        if len(v) == 1:
            return initialSplit
        else:
            pairs = createPairs(v)
            l = [splitSegment(x[0],x[1],n) for (x,n) in zip(pairs,npoints[1:])]
            return initialSplit + [item for sublist in l for item in sublist]
    ncol = _lambdas.shape[1]
    newVols = np.empty([np.sum(splitPoints)-1,ncol])
    for j in range(0,ncol):
        newVols[:,j] = vectorSplit(_lambdas[:,j],splitPoints)
    return newVols
   
def insertVols(maturities,_lambdas,maturity):#Check tomorrow
    def line(x,p1,p2):
        return p1[1]+(p2[1]-p1[1]) * (x-p1[0])/(p2[0]-p1[0])
    position = bisect.bisect_left(maturities,maturity)
    sublambda = _lambdas[(position-1):(position+1),:]
    firstTenor, secondTenor = maturity - maturities[position-1], maturities[position] - maturity
    nrow = _lambdas.shape[0]
    ncol = _lambdas.shape[1]
    newVols = np.empty([2,ncol])
    def volLine(x,l) :
        return line(x,(maturities[position]-1,l[0]),(maturities[position],l[1]))
    for j in range(0,ncol):
        lb = volLine(maturity,(sublambda[0,j],sublambda[1,j]))
        newVols[:,j] = np.array([lb,(sublambda[1,j] - firstTenor * lb)/secondTenor])
    newmaturities = np.concatenate((maturities[0:position],[maturity],maturities[position:len(maturities)]),0)
    return np.concatenate((_lambdas[0:position,:],newVols,_lambdas[(position+1):nrow,:]),0),newmaturities

def extendCalibration(initialMaturities,initialZCByieldcurve,principalComponents,Volatilities,splitPoints,extraMaturity):
    initialVolatilities = principalComponents * Volatilities #cal['Components'] * cal['Volatilities']
    originalMaturities = initialMaturities
    if extraMaturity != None and extraMaturity > initialMaturities[0]:
        initialVolatilities, initialMaturities = insertVols(initialMaturities,initialVolatilities,extraMaturity)
    initialVolatilities = np.concatenate((0.75 * initialVolatilities[0:1,:],initialVolatilities),0)#This step is only used to add the missing dimension
    _lambdas = np.concatenate((splitForwardVols(initialVolatilities[0:len(splitPoints),:],splitPoints),initialVolatilities[len(splitPoints):,:]),0)
    maturitiesToSplit = initialMaturities[0:len(splitPoints)]#(cal['Maturities'])[0:len(splitPoints)]
    zcbToSplit = np.concatenate((np.array([1.0]), initialZCByieldcurve),0)#[0:len(splitPoints)]
    l = [np.array([0.0])] + [(x - 1.0) + np.arange(1,y+1)/y for (x,y) in zip(maturitiesToSplit,splitPoints)]
    irregularmaturities = np.array([item for sublist in l for item in sublist])
    #print(irregularmaturities)
    #print(originalMaturities)
    allBonds = np.interp(np.concatenate((irregularmaturities,initialMaturities[len(splitPoints):]),0),np.concatenate(([0],maturitiesToSplit,originalMaturities[len(splitPoints):]),0),zcbToSplit)
    return {'ZCBInitialCurve':allBonds[1:],#np.concatenate((shortMaturityBonds[1:],initialZCByieldcurve[len(splitPoints):]))
            'Maturities':np.concatenate((irregularmaturities[1:],initialMaturities[len(splitPoints):])),
            'Components':principalComponents,'Volatilities':Volatilities,'_lambdas':_lambdas}
    
#import sklearn.decomposition
#p = sklearn.decomposition.PCA(3)
#pca =p.fit_transform(df)
#p.components_
#sdeviations = p.singular_values_() / df.shape[0]
#p.explained_variance_
#p.explained_variance_ratio_
#p.explained_variance_ratio_.cumsum()
#plt.plot(mydata)

#maturities =np.array(range(1,21),dtype=np.str)
