# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:47:31 2019

@author: Norberto
"""

import numpy as np#, quandl
import Black, copy
from sklearn import decomposition, linear_model#, preprocessing, feature_selection

#quandl.ApiConfig.api_key = "xiLjHtDE3ovNashy8puw"
#
#def writeFile(mydata, file = 'C:/Users/Norberto/Documents/StockData.csv') :
#    mydata.to_csv(file)
#
#def importFromQuandl(obs=500):
#    data = quandl.get(["WIKI/AMZN.4","WIKI/GM.4","WIKI/JPM.4","WIKI/KO.4","WIKI/MSFT.4"],  rows=obs)
#    data.dropna(inplace=True)
#    return data

def calcLogReturns(mydata):
    data = np.array(mydata.values,dtype=np.float_)
    return np.log(np.delete(data,(0),axis=0)) - np.log(np.delete(data,(data.shape[0]-1),axis=0))

def runStandardCalibration(mydata,delta = 1/252):
    returns = calcLogReturns(mydata)
    means = np.mean(returns,axis=0)
    correlation = np.corrcoef(returns,rowvar=False)
    returnVols = np.std(returns,axis=0)
    timescaling = 1.0 / delta
    return {'MeanReturns':means * timescaling * 0.25, 'Vols':returnVols *np.sqrt(timescaling), 'Correlation': correlation}
#Average historical returns are too high, hence we divide by 4 to make the calibration more realistic - no changes are made to the covariance structure

def runCalibration(mydata, nfactors=2, delta = 1 / 252):
    returns = calcLogReturns(mydata)
    returnVols = np.std(returns,axis=0)
    p = decomposition.PCA(returns.shape[1])
    p.fit_transform(returns)
    means = p.mean_
    biglambda = np.transpose(p.components_)
    vols = p.singular_values_ / np.sqrt(returns.shape[0])
    smalllambda = np.matmul(biglambda,np.diag(vols))
    #print(smalllambda)
    scalings = np.empty(returns.shape[1])
    for j in range (0,returns.shape[1]):
        scalings[j] = returnVols[j] / np.sqrt((smalllambda[j,0:nfactors] * smalllambda[j,0:nfactors]).sum())
        
    new_small_lambda = np.matmul(np.diag(scalings),smalllambda)
    timescaling = 1.0 / delta
    return {'MeanReturns':means * timescaling * 0.25, 'Vols':(new_small_lambda[:,0:nfactors]) * np.sqrt(timescaling), 'Correlation': np.diag(np.ones(nfactors))}
#Average historical returns are too high, hence we divide by 4 to make the calibration more realistic - no changes are made to the covariance structure
    
def gbm(s,drift,sigma,delta,variate):
    return s*np.exp((drift-0.5*sigma*sigma)*delta+np.sqrt(delta)*sigma*variate)

def gbmPath(s,mean,vols,corr,delta,horizon,sims):
    number_of_steps = int(horizon / delta)
    dimension = len(mean)
    simarray = []#[None] * sims#[]
    for i in range (0,sims):
        x = np.empty((len(s),number_of_steps + 1))
        x[:,0] = s
        y = s
        variates = np.random.multivariate_normal(np.zeros(dimension), corr, number_of_steps)
        for j in range (0,number_of_steps):
            y = gbm(y,mean,vols,delta,variates[j,:])
            x[:, j + 1 ] = y
        simarray.append(x)
        #simarray[i] = x
    return simarray

def gbmPathFromCalibration(calibration):
    averageReturns = calibration['MeanReturns']
    volatilities = calibration['Vols']
    correlation = calibration['Correlation']
    s = np.ones(len(averageReturns))
    return lambda d,h,n: gbmPath(s,averageReturns,volatilities,correlation,d,h,n)

def gbmPathFromFactors(s,means,vols,corr,delta,horizon,sims):
    number_of_steps = int(horizon / delta)
    dimension = corr.shape[0]
    drifts = means - 0.5 * np.array(list(map(lambda x :sum(x *x), vols)))
    simarray = []
    for i in range (0,sims):
        x = np.empty((len(s),number_of_steps + 1))
        x[:,0] = s
        y = s
        variates = np.random.multivariate_normal(np.zeros(dimension), corr, number_of_steps)
        for j in range (0,number_of_steps):
            y = y * np.exp(drifts * delta + np.sqrt(delta) * np.matmul(vols,np.transpose(variates[j,:])))
            x[:, j + 1 ] = y
        simarray.append(x)
    return simarray
        
def gbmPathFromFactorsAndCalibration(calibration):
    averageReturns = calibration['MeanReturns']
    volatilities = calibration['Vols']
    correlation = calibration['Correlation']
    s = np.ones(len(averageReturns))
    return lambda d,h,n: gbmPathFromFactors(s,averageReturns,volatilities,correlation,d,h,n)

def runSimulation(calibration,step = 0.5,horizon = 1.,number_of_paths = 10, isStandard=True):
    if isStandard:
        simulation = gbmPathFromCalibration(calibration)
    else:
        simulation = gbmPathFromFactorsAndCalibration(calibration)
    return simulation(step,horizon,number_of_paths)

def catchException(vols):
    try:
        vols = np.sqrt(np.array(list(map(lambda x :sum(x *x), vols))))
    except Exception:
        vols = vols    
    return vols
 
def createPositions(l,calibration):
    vols = calibration['Vols']
    vols = catchException(vols)
    equities = l['Equities']
    calls = l['Calls']
    puts = l['Puts']
    positions = []
    for j in range(0,len(equities)):
        if equities[j] != 0:
            positions.append((equities[j],lambda x,t: x,j))
        callpositions = list(filter(lambda y: y[0]==j and y[1]!=0,calls))
        for i in range(0,len(callpositions)):      
            q = callpositions[i][1]
            d = callpositions[i][2]
            m = callpositions[i][3]
            v = vols[j]
            positions.append((q,lambda x,t, q=q, d=d, m=m, v=v: Black.blackCall(x, 1.0, d,v,m-t),j))
        putpositions = list(filter(lambda y: y[0]==j and y[1]!=0,puts))
        for i in range(0,len(putpositions)): 
            q = putpositions[i][1]
            d = putpositions[i][2]
            m = putpositions[i][3]
            v = vols[j]
            positions.append((q,lambda x,t, q=q, d=d, m=m, v=v: Black.blackPut(x, 1.0, d,v,m-t),j))
    #print(positions[0][1](1,0),positions[1][1](1,0))
    return positions

def newInitialPortfolioValue(initialPositions):
    return sum(list(map(lambda x: x[0] * x[1](1,0),initialPositions)))

def newPortfolioValue(sim,initialPositions,t,delta):
    timestep = int(t/delta)
    valueDistribution = np.zeros(len(sim))
    for i in range (0,len(initialPositions)):
        scenarios = list(map(lambda x: x[initialPositions[i][2]][timestep],sim))
        valueDistribution += (initialPositions[i][0] * initialPositions[i][1](np.array(scenarios),t))
    return valueDistribution

def createPortfolioReturnDistribution(mydata,portfolio,stepsize,horizon,number_of_paths,isStandard=True):
    if isStandard:
        calibration = runStandardCalibration(mydata)
    else:
        calibration = runCalibration(mydata)
    simulation = runSimulation(calibration,stepsize,horizon,number_of_paths, isStandard)
    positions = createPositions(portfolio, calibration)
    return np.log(newPortfolioValue(simulation,positions,0.5,stepsize)/newInitialPortfolioValue(positions))

def performAMCregression(sim,initialPositions,t,delta):
    value_at_time_t = newPortfolioValue(sim,initialPositions,t,delta)
    previous_time_step = int(t/delta) - 1
    predictors = np.array(list(map(lambda x: x[:,previous_time_step],sim)))
    lm = linear_model.LinearRegression()
#    selector = feature_selection.RFE(lm,10)
#    poly = preprocessing.PolinomialFeatures(3)
#    result = selector.fit(poly.fit_transform(predictors),value_at_time_t.reshape(-1,1))
#    coeffs = result.estimator_.coef_
#    pm = lm.fit(poly.fit_transform(predictors),value_at_time_t.reshape(-1,1))
    x = lm.fit(predictors, value_at_time_t.reshape(-1,1))
    return x

#Previous code, still fully working but much slower
def createOldPositions(l,calibration):
    means = calibration['MeanReturns']
    vols = calibration['Vols']
    equities = l['Equities']
    calls = l['Calls']
    puts = l['Puts']
    positions = []
    for j in range(0,len(equities)):
        if equities[j] != 0:
            positions.append((equities[j],Black.Equity(1.0,means[j],vols[j]),j))
        callpositions = list(filter(lambda y: y[0]==j and y[1]!=0,calls))   
        for i in range(0,len(callpositions)):   
           positions.append((callpositions[i][1],Black.EquityOption(1.0, 1.0, callpositions[i][2],vols[j],callpositions[i][3],True),j)) 
        putpositions = list(filter(lambda y: y[0]==j and y[1]!=0,puts))
        for i in range(0,len(putpositions)): 
           positions.append((putpositions[i][1],Black.EquityOption(1.0, 1.0, putpositions[i][2],vols[j],putpositions[i][3],False),j))
    return positions        
   
def positionValue(p):
    return lambda t: p[0] * p[1].modelprice(p[1]._initialmaturity - t)
    #return lambda t: p[0] * p[1](p[1]._initialmaturity - t)

def initialPortfolioValue(initialPositions):
    return sum(list(map(lambda x: positionValue(x)(0),initialPositions)))

def portfolioValue(sim, initialpositions, t, delta):
    timestep = int(t/delta)
    valueDistribution = np.empty(len(sim))
    positions = copy.deepcopy(initialpositions)
    for j in range(0, len(sim)):
        scenario = (sim[j])[:,timestep]
        for i in range (0, len(positions)):
            (positions[i])[1]._price = scenario[positions[i][2]]
        valueDistribution[j] = sum(list(map(lambda x: positionValue(x)(t),positions)))
    return valueDistribution
        



        



























  