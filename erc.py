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
from numpy.testing import assert_almost_equal


def getERCWeights(cov, x0=None, options=None, scale=10000):

    # checking cov is pd
    if np.any(np.linalg.eigvals(cov) <= 0):
        return None
    if options == None:
        options = {'ftol':1e-20, 'maxiter':2000, 'disp':True}

    n = cov.shape[0]

    # naive inverse weighting
    if x0 == None:
        diagInv = 1/np.sqrt(np.diag(cov))
        x0 = diagInv/diagInv.sum()

    # per Maillard et al. (2008)
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
    # t0: priced_dt datetime object
    # ret: full return series
    # numOfMons: number of months (bwd) for covariance estimation
    tminusN = t0 + relativedelta(months=-numOfMons)
    relevantRet = ret[tminusN:t0]
    cov = np.cov(relevantRet.values, rowvar=False)
    return cov

#
def loopGetERCWeights(ret, x0=None, options=None, scale=10000, numOfMons=12):
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

df = pd.read_csv(r'C:\Users\Mike Song\Google Drive\Balanced Portfolio\smallDataset-test.csv', index_col=0, parse_dates=[0])

#A = np.random.rand(10, 10)/100
#cov = A.dot(A.transpose())
#X = getERCWeights(cov)
#mrcRatio = np.dot(cov, X)*X/np.dot(np.dot(X.T, cov), X)



loopWeights, list = loopGetERCWeights(df)