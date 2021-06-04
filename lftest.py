#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 02:33:15 2020

@author: cwlab08
"""

import os
import rempropensity as rp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

gmmDF = pd.read_csv('2gmmDF.csv')
linfitDF = pd.read_csv('2linfitDF.csv')
logfitDF = pd.read_csv('2logfitDF.csv')


klow = gmmDF.klow
khigh = gmmDF.khigh
mlow = gmmDF.mlow
mhigh = gmmDF.mhigh
slow = gmmDF.slow
shigh = gmmDF.shigh

ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

remDF = rp.standard_recToDF(ppath, recordings, 8, False)
remDF['loginter'] = remDF['inter'].apply(np.log)
remDF = remDF.loc[remDF.rem<240]


splitDFs = rp.splitData(remDF,30)

ks_sims = []
ks_obs = []

from scipy.stats import rv_continuous
xvals = np.arange(0,10,0.01)

for idx,subDF in enumerate(splitDFs):
    kssim = []
    subdata = list(subDF.logsws)
    k1 = khigh[idx]
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh[idx]
    s1 = shigh[idx]
    m2 = mlow[idx]
    s2 = slow[idx]
    
    class mix_gaussian(rv_continuous):
        def _pdf(self, x):
            if k2==0:
                return k1*stats.norm.pdf(x,m1,s1)
            else:
                return k1*stats.norm.pdf(x,m1,s1) + k2*stats.norm.pdf(x,m2,s2)
    
    mygauss = mix_gaussian()
    cdfvalues = mygauss.cdf(xvals)
    drawsize = len(subdata)
    
    for i in range(10000):
        np.random.seed(i)
        rand_draws = np.random.uniform(0,1,drawsize)
        samples = []
        for x in rand_draws:
            criticalPlace = np.where(cdfvalues>=x)[0]
            if len(criticalPlace>0):
                sampleIndex = min(criticalPlace)
            else:
                sampleIndex = len(xvals)-1
            sampleVal = xvals[sampleIndex]
            samples.append(sampleVal)
        if k2==0:
            a = stats.kstest(samples, lambda x:k1*stats.norm.cdf(x,m1,s1))
        else:
            a = stats.kstest(samples, lambda x:k1*stats.norm.cdf(x,m1,s1)+k2*stats.norm.cdf(x,m2,s2))
        ks = a[0]
        kssim.append(ks)
        print(str(idx) + '---' + str(i))
    if k2==0:
        ksobs = stats.kstest(subdata, lambda x:k1*stats.norm.cdf(x,m1,s1))
    else:
        ksobs = stats.kstest(subdata, lambda x:k1*stats.norm.cdf(x,m1,s1)+k2*stats.norm.cdf(x,m2,s2))
    ks_sims.append(kssim)
    ks_obs.append(ksobs)

teststats = [x[0] for x in ks_obs]

pvals = []
for idx,kssim in enumerate(ks_sims):
    ksobs = teststats[idx]
    pval = (sum(kssim>ksobs))/(10000)
    pvals.append(pval)

fig1 = plt.figure()
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for idx,x in enumerate(ks_sims):
    plt.subplot(2,4,idx+1)
    plt.title(str(30*idx)+'<rem<'+str(30*(idx+1)))
    plt.hist(x, bins=20, color='gray', alpha=0.7, label='Sim. dist')
    plt.axvline(x=teststats[idx], color='red', label='observed')
    plt.xlabel('KS-stat')
    plt.ylabel('# trials')
    if idx==7:
        plt.legend()
    sns.despine()

lfDF = pd.DataFrame(list(zip(teststats,pvals)),columns=['tstat','pval'])
lfDF.to_csv("lilliefors.csv")

