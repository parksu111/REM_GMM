#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:15:50 2020

@author: cwlab08
"""

import numpy as np
import os.path
import matplotlib.pylab as plt
import pandas as pd
import rempropensity as rp
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm


ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

darkDF = rp.standard_recToDF(ppath, recordings, 8, True)
darkDF = darkDF.loc[darkDF.rem<240]


splitDFs = rp.splitData(darkDF, 30)
splitLogSws = [x.logsws for x in splitDFs]


klow = []
khigh = []
mlow = []
mhigh = []
slow = []
shigh = []

for idx,data in enumerate(splitLogSws):
    if (idx<3)or(idx==5):
        gmm = GaussianMixture(n_components=2, tol=0.000001, max_iter=999999999)
        gmm.fit(np.expand_dims(data,1))
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        k1 = weights[0]
        k2 = weights[1]
        m1 = means[0]
        m2 = means[1]
        s1 = stds[0]
        s2 = stds[1]
        k1 = np.round(k1,6)
        k2 = np.round(k2,6)
        m1 = np.round(m1,6)
        m2 = np.round(m2,6)
        s1 = np.round(s1,6)
        s2 = np.round(s2,6)
        if m1<m2:
            klow.append(k1)
            khigh.append(k2)
            mlow.append(m1)
            mhigh.append(m2)
            slow.append(s1)
            shigh.append(s2)
        else:
            klow.append(k2)
            khigh.append(k1)
            mlow.append(m2)
            mhigh.append(m1)
            slow.append(s2)
            shigh.append(s1)
    else:
        gmm = GaussianMixture(n_components=1, tol=0.000001, max_iter=999999999)
        gmm.fit(np.expand_dims(data,1))
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        k = weights[0]
        m = means[0]
        s = stds[0]
        k = np.round(k,6)
        m = np.round(m,6)
        s = np.round(s,6)
        khigh.append(k)
        mhigh.append(m)
        shigh.append(s)
        klow.append(None)
        mlow.append(None)
        slow.append(None)


def log_func(x,a,b,c):
    return a*np.log(x+b)+c

xes = np.arange(15,240,30)
lowxes = [15,45,75,165]

amlow = [mlow[0],mlow[1],mlow[2],mlow[5]]
aslow = [slow[0],slow[1],slow[2],slow[5]]

khigh_log = curve_fit(log_func, xes[0:4], khigh[0:4], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]))[0]
khigh_loga = khigh_log[0]
khigh_logb = khigh_log[1]
khigh_logc = khigh_log[2]

mhigh_log = curve_fit(log_func, xes, mhigh,bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]))[0]
mhigh_loga = mhigh_log[0]
mhigh_logb = mhigh_log[1]
mhigh_logc = mhigh_log[2]

mlow_log = curve_fit(log_func, lowxes, amlow, p0=[-1,1,1], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]))[0]
mlow_loga = mlow_log[0]
mlow_logb = mlow_log[1]
mlow_logc = mlow_log[2]

shigh_log = curve_fit(log_func, xes, shigh, p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
shigh_loga = shigh_log[0]
shigh_logb = shigh_log[1]
shigh_logc = shigh_log[2]

slow_log = curve_fit(log_func, lowxes, aslow, p0=[-1,1,1], bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
slow_loga = slow_log[0]
slow_logb = slow_log[1]
slow_logc = slow_log[2]


xplot1 = np.arange(0,240)
xplot2 = np.arange(0,160)


#khigh
x1 = xes[0:7]
x1 = sm.add_constant(x1)
y1 = khigh[0:7]
mod1 = sm.OLS(y1,x1)
res1 = mod1.fit()
linreg1 = lambda x: res1.params[1]*x+res1.params[0]

#mhigh
x2 = xes
x2 = sm.add_constant(x2)
y2 = mhigh
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()
linreg2 = lambda x: res2.params[1]*x+res2.params[0]

#mlow
x3 = lowxes
x3 = sm.add_constant(x3)
y3 = amlow
mod3 = sm.OLS(y3,x3)
res3 = mod3.fit()
linreg3 = lambda x: res3.params[1]*x+res3.params[0]

#shigh
x4 = xes
x4 = sm.add_constant(x4)
y4 = shigh
mod4 = sm.OLS(y4,x4)
res4 = mod4.fit()
linreg4 = lambda x: res4.params[1]*x+res4.params[0]

#slow
x5 = lowxes
x5 = sm.add_constant(x5)
y5 = aslow
mod5 = sm.OLS(y5,x5)
res5 = mod5.fit()
linreg5 = lambda x: res5.params[1]*x+res5.params[0]

khigh_lin=[res1.params[1],res1.params[0]]
mhigh_lin=[res2.params[1],res2.params[0]]
mlow_lin=[res3.params[1],res3.params[0]]
shigh_lin=[res4.params[1],res4.params[0]]
slow_lin=[res5.params[1],res5.params[0]]


gmmDF = pd.DataFrame(list(zip(khigh,mhigh,mlow,shigh,slow)),columns=['khigh','mhigh','mlow','shigh','slow'])
logfitDF = pd.DataFrame(list(zip(khigh_log,mhigh_log,mlow_log,shigh_log,slow_log)),columns=['khigh','mhigh','mlow','shigh','slow'])
linfitDF = pd.DataFrame(list(zip(khigh_lin,mhigh_lin,mlow_lin,shigh_lin,slow_lin)),columns=['khigh','mhigh','mlow','shigh','slow'])

gmmDF.to_csv('darkgmmDF.csv')
logfitDF.to_csv('darklogfitDF.csv')
linfitDF.to_csv('darklinfitDF.csv')

