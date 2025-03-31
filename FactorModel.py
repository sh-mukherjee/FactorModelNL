# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:58:08 2019

@author: Norberto
"""

import numpy as np, pandas, AMC
stockDataFile = '/workspaces/FactorModelNL/FTSE_stocks.csv'

capsdivsDataFile = '/workspaces/FactorModelNL/CapsDivs.csv'

sectorDataFile = '/workspaces/FactorModelNL/Sectors.csv'

def readFile(filename,drop):
    data = pandas.read_csv(filename)
    if drop==True:
        data.dropna(inplace=True)
    return data

def importData(files):
    alldata = [readFile(f,d) for (f,d) in zip(files,[True,False,True])]
    return {'stocks':alldata[0],'capsdivs':(alldata[1]).drop('TotalMarketCap',axis=1),'sectors':alldata[2],'marketcap':(alldata[1])['TotalMarketCap'][0]}

def weightsFromCaps(capsDivs,totalCap):
    caps = capsDivs['Cap']
    return caps/totalCap

def sectorBetas(sectornames):
    #s = list(zip(sectornames,sectornames.values[0]))
    #s.sort(key = lambda x: x[0])
    #sectors = np.unique([x[1] for x in s])
    sectornames.sort_index(axis=1,inplace=True)
    sectors = np.unique(sectornames)
    def betafuncs(z):
        return [1.0 if z==x else 0.0 for x in sectors]
    #betas = [betafuncs(z) for z in [x[1] for x in s]]
    betas = sectornames.applymap(betafuncs).T.iloc[:,0].apply(pandas.Series)
    betas.columns = sectors
    return np.array(betas), sectors
#[stockDataFile,capsdivsDataFile,sectorDataFile]
def normaliseVariables(variable,allstocknames,stocknames):
    logvariable = np.log(np.array(variable,dtype=np.float))
    oklogvariable = np.nan_to_num(logvariable)#logvariable[~np.isnan(logvariable)]
    logvariableWithNames = [(l,n) for (l,n) in zip(logvariable,allstocknames) if ~np.isnan(l)]
    m, sd = np.mean(oklogvariable), np.std(oklogvariable,ddof=1)
    normalisedWithNames = [((l[0]-m)/sd,l[1]) for l in logvariableWithNames]
    normalisedWithNames = list(filter(lambda y: y[1] in stocknames, normalisedWithNames))
    normalisedWithNames.sort(key = lambda x: x[1])
    return normalisedWithNames
#normaliseVariables(data['capsdivs'][:,1],data['capsdivs'][:,0],(data['stocks']).columns)
    #normWeightedVariables(cd['Div'],cd['Company'],(data['stocks']).columns,np.array(myweights))
def normWeightedVariables(variable,allstocknames,stocknames,capweights):
    variable = np.array(variable,dtype=np.float)
    okvariable, okweights = variable[~np.isnan(variable)], capweights[~np.isnan(variable)]
    variableWithNames = [(l,n) for (l,n) in zip(variable,allstocknames) if ~np.isnan(l)]
    m, sd = np.average(okvariable,weights=okweights), np.sqrt(np.cov(okvariable,aweights=okweights))
    normalisedWithNames = [((l[0]-m)/sd,l[1]) for l in variableWithNames]
    normalisedWithNames = list(filter(lambda y: y[1] in stocknames, normalisedWithNames))
    normalisedWithNames.sort(key = lambda x: x[1])
    return normalisedWithNames

def logReturns(values,days):
    return np.log(values[days:,:]) - np.log(values[0:(values.shape[0]-days),:])
    #values.diff(1).drop(0)
def excessReturns(stocks):#stocks=alldata[0]#stocks =stocks.drop('Date',axis=1)
    rs = logReturns(np.array(stocks),1)
    return rs[:,1:] - rs[:,0].reshape(rs.shape[0],1)
#stocks.drop(['.FTSE'],axis=1).apply(lambda x : x - stocks['.FTSE'])
def calculateOLSFactors(excessrets,normalisedBetas,industrysectors):
    allbetas = np.concatenate((np.transpose(np.array([list(zip(*b))[0] for b in normalisedBetas])),industrysectors),axis=1)
    #tbetasinverse = np.linalg.inv(np.matmul(np.transpose(allbetas),allbetas))
    olsbetas = np.matmul(allbetas,np.linalg.inv(np.matmul(np.transpose(allbetas),allbetas)))#np.matmul(tbetasinverse,np.transpose(allbetas))
    estimatedFactors = np.matmul(excessrets,olsbetas)#np.transpose(olsbetas)
    residuals = excessrets - np.matmul(estimatedFactors,np.transpose(allbetas))
    olscov = np.cov(estimatedFactors.T,ddof=1)
    olsresvariance = np.var(residuals,axis=0,ddof=1) #np.diag(np.cov(residuals,ddof=1))
    return {'Betas':allbetas,'OLSfactors':estimatedFactors,'OLScov':olscov,'OLSresiduals':residuals,'Weights':olsresvariance}

def calculateWLSFactors(excessrets,OLSoutput):
    weightmatrix = np.diag(1/OLSoutput['Weights'])
    betas = OLSoutput['Betas']
    weightedBetasInverse = np.linalg.inv(np.matmul(np.matmul(np.transpose(betas),weightmatrix),betas))
    wlsBetas = np.matmul(np.matmul(weightedBetasInverse,np.transpose(betas)),weightmatrix)
    wlsFactors = np.matmul(excessrets,np.transpose(wlsBetas))
    residuals = excessrets - np.matmul(wlsFactors,np.transpose(betas))
    wlscov = np.cov(wlsFactors.T,ddof=1)
    wlsmeans = np.mean(wlsFactors,axis=0)
    resvariance = np.var(residuals,axis=0,ddof=1)
    resmeans = np.mean(residuals,axis=0)
    return {'Betas':betas,'WLSfactors':wlsFactors,'WLSmean':wlsmeans,'WLScov':wlscov,'WLSresiduals':residuals,'Residualmeans':resmeans,'Residualvariances':resvariance}
#[normWeightedVariables(v,cd['Company'],(data['stocks']).columns,np.array(myweights)) for v in np.transpose(ll).tolist()]
def calibrateFactorModel():
    mydata = importData([stockDataFile,capsdivsDataFile,sectorDataFile])
    cd = mydata['capsdivs']    
    myweights = np.array(weightsFromCaps(cd,mydata['marketcap']))
    weightNormalisedBetas = normWeightedVariables(cd['Div'],cd['Company'],(mydata['stocks']).columns,myweights)
    weightnames = ['Div'] 
    l = np.array(cd.drop(['Company','Div'],axis=1))
    regularNormalisedBetas = [normaliseVariables(v,cd['Company'],(mydata['stocks']).columns) for v in np.transpose(l).tolist()]
    regularnames = set(cd.columns)-set(['Company','Div'])
    normBetas = [x for s in [[weightNormalisedBetas],regularNormalisedBetas] for x in s]
    sBetas, sectornames = sectorBetas(mydata['sectors'])
    excessrets = excessReturns((mydata['stocks']).drop('Date',axis=1))
    ols = calculateOLSFactors(excessrets,normBetas,sBetas)
    wls = calculateWLSFactors(excessrets,ols)
    allnames = np.concatenate((weightnames,list(regularnames),sectornames),axis=0)
    stocknames = list(mydata['sectors'])
    stocknames.sort()
    wls['Betas'] = pandas.DataFrame(wls['Betas'],columns = allnames, index = stocknames)
    return wls

def simulateFactorModel(calibration):
    meanreturns = calibration['Betas']@calibration['WLSmean'] + calibration['Residualmeans']
    annualMeans = meanreturns * 260
    annualCov = calibration['WLScov'] * 260
    annualVols = np.sqrt(np.diag(annualCov))
    corr = np.corrcoef(annualCov)
    volstructure = np.array(calibration['Betas']@np.diag(annualVols))
    factorcalibration = {'MeanReturns':annualMeans,'Vols': volstructure,'Correlation':corr}
    sim = AMC.runSimulation(factorcalibration,1.,1.,10,False)
    simAtStep = np.array([x[:,1] for x in sim])
    return simAtStep
#cd = data['capsdivs']
#y =cd.drop(['Company','Div'],axis=1)
#l=np.array(y)
# nv = [normaliseVariables(v,cd['Company'],(data['stocks']).columns) for v in np.transpose(l).tolist()]
