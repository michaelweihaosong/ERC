# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:07:13 2018

@author: Mike Song
"""
import datetime
from dateutil.relativedelta import relativedelta
import scipy.optimize as opt
import numpy as np
import pandas as pd


def getERCWeights(cov, x0=None, options=None, scale=10000):
    """
    Get optimal weights such that each asset contributes equal risk to the portfolio.
    Implementation is according to Maillard et al. (2008)
    
    ------
    param
    ------
    cov | ndarray
        covariance matrix of the assets in the portfolio
    x0 | ndarray
        initial weights for optimization
    options | dictionary
        a dictionary containing options for the optimization
    scale | float
        scaling number used to help convergence of the optimization
    
    ------
    return
    ------
    X | ndarray
        optimal weights of the equal risk contribution portfolio
    mrcRatio | ndarray
        an ndarray representing the risk contribution of each asset (in terms of percentage)
    """
    # checking cov is pd
    if np.any(np.linalg.eigvals(cov) <= 0):
        return None
    # scaling cov if F-Norm is too small
    if np.linalg.norm(cov) <= 0.1:
        cov = cov*scale
    if options == None:
        options = {'ftol':1e-20, 'maxiter':2000, 'disp':True}

    n = cov.shape[0]

    # naive inverse weighting
    if x0 == None:
        diagInv = 1/np.sqrt(np.diag(cov))
        x0 = diagInv/diagInv.sum()

    def obj(x):
        riskCon = np.dot(cov, x)*x
        riskCon = np.reshape(riskCon, (len(x), 1))
        diffMat = riskCon - riskCon.T
        cost = np.linalg.norm(diffMat)**2
        return cost/scale

    bounds = [(0, 1) for i in range(n)]
    constraints = {'type': 'eq', 'fun': lambda x: sum(x)-1}

    optres = opt.minimize(obj, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options=options)

    X = optres.x
    mrcRatio = np.dot(cov, X)*X/np.dot(np.dot(X.T, cov), X)
    return X, mrcRatio


def retToCov(t0, ret, numOfMons = 12):
    """
    get a covariance matrix from returns
    
    ------
    param
    ------
    t0 | datetime
        datetime object of the pricing date
    ret | pandas.dataframe
        dataframe containing the entire return series
    numOfMons | float
        number representing how long is the estimation window for covariance matrix
        
    ------
    return
    ------
    cov | ndarray
        covariance matrix
    """
    # t0: priced_dt datetime object
    # ret: full return series
    # numOfMons: number of months (bwd) for covariance estimation
    tminusN = t0 + relativedelta(months=-numOfMons)
    relevantRet = ret[tminusN:t0]
    cov = np.cov(relevantRet.values, rowvar=False)
    return cov

#
def loopGetERCWeights(ret, x0=None, options=None, scale=10000, numOfMons=12):
    """
    wrapper function used to iterate over the entire return series to calculate optimal weights at each pricing date
    
    ------
    param
    ------
    ret | pandas.dataframe
        dataframe containing the entire return series
    x0 | ndarray
        initial weights for optimization
    options | dictionary
        a dictionary containing options for the optimization
    scale | float
        scaling number used to help convergence of the optimization
    numOfMons | float
        number representing how long is the estimation window for covariance matrix
    
    ------
    return
    ------
    retval | pandas.dataframe
        dataframe containing the optimal weights for each pricing date from the input
    mrcRatioList | list
        list containing the ndarrays representing the risk contribution of each asset (in terms of percentage) for each pricing date from the input
    """
    retVal = ret.copy()
    mrcRatioList = []
    
    for date in ret.index:
        
        tminusN = date + relativedelta(months=-numOfMons)
        if tminusN < min(ret.index):
            retVal.loc[date] = np.full((ret.shape[1], ), np.nan)
        else:
            cov = retToCov(date, ret, numOfMons=numOfMons)
            X, mrcRatio = getERCWeights(cov, x0=x0, options=options, scale=scale)
            retVal.loc[date] = X
            mrcRatioList.append(mrcRatio)
        
    return retVal, mrcRatioList

