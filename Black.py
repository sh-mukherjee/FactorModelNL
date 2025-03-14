# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:49:02 2019

@author: Norberto
"""
import numpy as np
from scipy.stats import norm

class Equity:
    def __init__(self,price,drift,volatility):
        self._price = price
        self._drift = drift
        self._volatility = volatility
        self._initialmaturity = 0
        
    def modelprice(self,maturity):
        return self._price
        
class EquityOption:
    def __init__(self,price,strike,drift,volatility,initialmaturity,iscall):
        self._price = price
        self._strike = strike
        self._drift = drift
        self._volatility = volatility
        self._initialmaturity = initialmaturity
        self._iscall = iscall
        
    def modelprice(self,maturity):
        if self._iscall == True:
            return blackCall(self._price,self._strike,self._drift,self._volatility,maturity)
        else :
            return blackPut(self._price,self._strike,self._drift,self._volatility,maturity)

def blackCall(s,k,r,sigma,maturity):
    if maturity > 0:
        stsqrt = sigma*np.sqrt(maturity)
        d1 = (np.log(s/k)+(r+0.5*sigma*sigma)*maturity)/stsqrt
        d2 = d1-stsqrt
        return s*norm.cdf(d1)-k*np.exp(-r*maturity)*norm.cdf(d2)
    else:
        return np.maximum(s - k,0)

def forward(s,k,r,maturity):
  return s-k*np.exp(-r*maturity)

def blackPut(s,k,r,sigma,maturity):
    if maturity > 0:
        return blackCall(s,k,r,sigma,maturity)-forward(s,k,r,maturity)
    else:
        return np.maximum(k - s,0)