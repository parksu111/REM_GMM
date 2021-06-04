#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:04:36 2020

@author: cwlab08
"""
#### Preparations

#Import necessary packages
import numpy as np
import os.path
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import rempropensity as rp
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Make dataframe containing all REM-NREM cycles
remDF = rp.standard_recToDF(ppath,recordings,8,False)
remDF = remDF.loc[remDF.rem<240]

###############################################################################
### Estimate GMM parameters


#Split dataframe by 30 seconds of REMpre
splitDFs = rp.splitData(remDF, 30)
splitLogSws = [x.logsws for x in splitDFs]

#Set seed
np.random.seed(1)

#Lists to store parameters
klow = []
khigh = []
mlow = []
mhigh = []
slow = []
shigh = []

#Go through each 30s split and estimate GMM parameters
for idx,data in enumerate(splitLogSws):
    if idx<5:
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

#Plot gmm parameters

xes = np.arange(15,240,30)

plt.figure()
plt.subplot(2,3,1)
plt.scatter(xes, khigh, s=10)
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,2)
plt.scatter(xes, mhigh, s=10)
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,3)
plt.scatter(xes, shigh, s=10)
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,5)
plt.scatter(xes, mlow, s=10)
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,6)
plt.scatter(xes, slow, s=10)
plt.xlim([0,240])
sns.despine()

###############################################################################
### Find log-model coefficients

#Define log function to model change of parameters
def log_func(x,a,b,c):
    return a*np.log(x+b)+c

#Fit coefficients of log function to each parameter
klow_log = curve_fit(log_func, xes[0:5], klow[0:5],p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]))[0]
klow_loga = klow_log[0]
klow_logb = klow_log[1]
klow_logc = klow_log[2]

khigh_log = curve_fit(log_func, xes[0:6], khigh[0:6], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]))[0]
khigh_loga = khigh_log[0]
khigh_logb = khigh_log[1]
khigh_logc = khigh_log[2]

mlow_log = curve_fit(log_func, xes[0:5], mlow[0:5], p0=[-1,1,1], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]))[0]
mlow_loga = mlow_log[0]
mlow_logb = mlow_log[1]
mlow_logc = mlow_log[2]

mhigh_log = curve_fit(log_func, xes, mhigh,bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]))[0]
mhigh_loga = mhigh_log[0]
mhigh_logb = mhigh_log[1]
mhigh_logc = mhigh_log[2]

slow_log = curve_fit(log_func, xes[0:5], slow[0:5], p0=[-1,1,1], bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
slow_loga = slow_log[0]
slow_logb = slow_log[1]
slow_logc = slow_log[2]

shigh_log = curve_fit(log_func, xes, shigh, p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
shigh_loga = shigh_log[0]
shigh_logb = shigh_log[1]
shigh_logc = shigh_log[2]


#Fit coefficients of linear function to each parameter
xplot1 = np.arange(0,240)
xplot2 = np.arange(0,160)

#klow
x1 = np.arange(15,240,30)[0:5]
x1 = sm.add_constant(x1)
y1 = klow[0:5]
mod1 = sm.OLS(y1,x1)
res1 = mod1.fit()
linreg1 = lambda x: res1.params[1]*x + res1.params[0]

#mlow
x2 = np.arange(15,240,30)[0:5]
x2 = sm.add_constant(x2)
y2 = mlow[0:5]
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()
linreg2 = lambda x: res2.params[1]*x + res2.params[0]

#mlow
x3 = np.arange(15,240,30)[0:5]
x3 = sm.add_constant(x3)
y3 = slow[0:5]
mod3 = sm.OLS(y3,x3)
res3 = mod3.fit()
linreg3 = lambda x: res3.params[1]*x + res3.params[0]

#khigh
x4 = np.arange(15,240,30)[0:6]
x4 = sm.add_constant(x4)
y4 = khigh[0:6]
mod4 = sm.OLS(y4,x4)
res4 = mod4.fit()
linreg4 = lambda x: res4.params[1]*x + res4.params[0]

#mhigh
x5 = np.arange(15,240,30)
x5 = sm.add_constant(x5)
y5 = mhigh
mod5 = sm.OLS(y5,x5)
res5 = mod5.fit()
linreg5 = lambda x: res5.params[1]*x + res5.params[0]

#shigh
x6 = np.arange(15,240,30)
x6 = sm.add_constant(x6)
y6 = shigh
mod6 = sm.OLS(y6,x6)
res6 = mod6.fit()
linreg6 = lambda x: res6.params[1]*x + res6.params[0]


#Plot gmm parameters along with log and linear function fits
plt.figure()
plt.subplot(2,3,1)
plt.title('k_high')
plt.scatter(xes, khigh, s=10)
plt.plot(xplot1,khigh_loga*np.log(xplot1+khigh_logb)+khigh_logc)
plt.plot(xplot1,linreg4(xplot1))
plt.xlim([0,240])
plt.ylim([0,1.1])
sns.despine()
plt.subplot(2,3,2)
plt.title('k_high')
plt.scatter(xes, mhigh, s=10)
plt.plot(xplot1,mhigh_loga*np.log(xplot1+mhigh_logb)+mhigh_logc)
plt.plot(xplot1,linreg5(xplot1))
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,3)
plt.title('s_high')
plt.scatter(xes, shigh, s=10)
plt.plot(xplot1,shigh_loga*np.log(xplot1+shigh_logb)+shigh_logc)
plt.plot(xplot1,linreg6(xplot1))
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,4)
plt.title('k_low')
plt.scatter(xes, klow, s=10)
plt.plot(xplot2, klow_loga*np.log(xplot2+klow_logb)+klow_logc)
plt.plot(xplot2,linreg1(xplot2))
plt.xlim([0,240])
plt.ylim([0,1.1])
sns.despine()
plt.subplot(2,3,5)
plt.title('m_low')
plt.scatter(xes, mlow, s=10)
plt.plot(xplot2, mlow_loga*np.log(xplot2+mlow_logb)+mlow_logc)
plt.plot(xplot2,linreg2(xplot2))
plt.xlim([0,240])
plt.ylim([3,6])
sns.despine()
plt.subplot(2,3,6)
plt.title('s_low')
plt.scatter(xes, slow, s=10)
plt.plot(xplot2, slow_loga*np.log(xplot2+slow_logb)+slow_logc)
plt.plot(xplot2,linreg3(xplot2))
plt.xlim([0,240])
sns.despine()

###############################################################################
### Save GMM parameters and model coefficients to a csv

khigh_lin = [res4.params[1],res4.params[0]]
klow_lin = [res1.params[1],res1.params[0]]
mhigh_lin = [res5.params[1],res5.params[0]]
mlow_lin = [res2.params[1],res2.params[0]]
shigh_lin = [res6.params[1],res6.params[0]]
slow_lin = [res3.params[1],res3.params[0]]

gmm2DF = pd.DataFrame(list(zip(khigh,klow,mhigh,mlow,shigh,slow)),columns=['khigh','klow','mhigh','mlow','shigh','slow'])
logfit2DF = pd.DataFrame(list(zip(khigh_log,klow_log,mhigh_log,mlow_log,shigh_log,slow_log)),columns=['khigh','klow','mhigh','mlow','shigh','slow'])
linfit2DF = pd.DataFrame(list(zip(khigh_lin,klow_lin,mhigh_lin,mlow_lin,shigh_lin,slow_lin)),columns=['khigh','klow','mhigh','mlow','shigh','slow'])

gmm2DF.to_csv('2gmmDF.csv')
logfit2DF.to_csv('2logfitDF.csv')
linfit2DF.to_csv('2linfitDF.csv')

###############################################################################