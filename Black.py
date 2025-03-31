# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:49:02 2019

@author: Norberto
"""
import numpy as np
from scipy.stats import norm # Importing the normal distribution for calculating option prices

class Equity:
    """
    Represents an equity (stock) with an initial price, drift (expected return), and volatility.
    """
    def __init__(self,price,drift,volatility):
        self._price = price
        self._drift = drift
        self._volatility = volatility
        self._initialmaturity = 0
        
    def modelprice(self,maturity):
        """
        Returns the price of the equity at any maturity (which is constant in this simple model).
        """
        return self._price
        
class EquityOption:
    """
    Represents an equity option (either call or put) with Black-Scholes pricing.
    """
    def __init__(self,price,strike,drift,volatility,initialmaturity,iscall):
        self._price = price # Underlying stock price
        self._strike = strike # Strike price of the option
        self._drift = drift # Risk-free rate (assumed as drift here)
        self._volatility = volatility # Volatility of the underlying asset
        self._initialmaturity = initialmaturity # Time to maturity (in years)
        self._iscall = iscall # Boolean: True for call, False for put
        
    def modelprice(self,maturity):
        """
        Computes the theoretical price of the option using the Black-Scholes model.
        """
        if self._iscall == True:
            return blackCall(self._price,self._strike,self._drift,self._volatility,maturity)
        else :
            return blackPut(self._price,self._strike,self._drift,self._volatility,maturity)

def blackCall(s,k,r,sigma,maturity):
    """
    Computes the Black-Scholes price for a European call option.
    """
    if maturity > 0:
        stsqrt = sigma*np.sqrt(maturity) # Standard deviation over time
        d1 = (np.log(s/k)+(r+0.5*sigma*sigma)*maturity)/stsqrt
        d2 = d1-stsqrt
        return s*norm.cdf(d1)-k*np.exp(-r*maturity)*norm.cdf(d2) # Black-Scholes formula
    else:
        return np.maximum(s - k,0) # Intrinsic value at maturity

def forward(s,k,r,maturity):
    """
    Computes the forward contract price (difference between spot and discounted strike).
    """
    return s-k*np.exp(-r*maturity)

def blackPut(s,k,r,sigma,maturity):
    """
    Computes the Black-Scholes price for a European put option using put-call parity.
    """
    if maturity > 0:
        return blackCall(s,k,r,sigma,maturity)-forward(s,k,r,maturity) # Using put-call parity
    else:
        return np.maximum(k - s,0) # Intrinsic value at maturity
