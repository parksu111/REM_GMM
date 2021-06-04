#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:18:50 2021

@author: cwlab08
"""

#### Preparations

#Import necessary packages
import os
import rempropensity as rp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.optimize import fsolve
from scipy import stats
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.integrate import quad

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/nrev_figs/ma0/'

#Define colors
current_palette = sns.color_palette('muted', 10)
col1 = current_palette[3]
col2 = current_palette[0]

#Make dataframe containing all REM-NREM cycles
remDF = rp.standard_recToDF(ppath, recordings, 8, False)
remDF = remDF.loc[remDF.rem<240]

#Read in GMM parameters and model coefficients
gmmDF = pd.read_csv('2gmmDF.csv')
linfitDF = pd.read_csv('2linfitDF.csv')
logfitDF = pd.read_csv('2logfitDF.csv')

klow = gmmDF.klow
khigh = gmmDF.khigh
mlow = gmmDF.mlow
mhigh = gmmDF.mhigh
slow = gmmDF.slow
shigh = gmmDF.shigh

klow_loga = logfitDF.klow[0]
klow_logb = logfitDF.klow[1]
klow_logc = logfitDF.klow[2]
khigh_loga = logfitDF.khigh[0]
khigh_logb = logfitDF.khigh[1]
khigh_logc = logfitDF.khigh[2]
mlow_loga = logfitDF.mlow[0]
mlow_logb = logfitDF.mlow[1]
mlow_logc = logfitDF.mlow[2]
mhigh_loga = logfitDF.mhigh[0]
mhigh_logb = logfitDF.mhigh[1]
mhigh_logc = logfitDF.mhigh[2]
slow_loga = logfitDF.slow[0]
slow_logb = logfitDF.slow[1]
slow_logc = logfitDF.slow[2]
shigh_loga = logfitDF.shigh[0]
shigh_logb = logfitDF.shigh[1]
shigh_logc = logfitDF.shigh[2]
slow_lina = linfitDF.slow[0]
slow_linb = linfitDF.slow[1]

###############################################################################

recordings = os.listdir(ppath)

rems30 = []
nrems30 = []
wakes30 = []

rems10 = []
nrems10 = []
wakes10 = []

rems0 = []
nrems0 = []
wakes0 = []


for rec in recordings:
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    Mtups30 = []
    Mtups10 = []
    Mtups0 = []
    for idx,x in enumerate(recs):
        Mvec = rp.nts(x)
        if idx==0:
            Mtup = rp.vecToTup(Mvec, start=0)
            fixtup30 = rp.ma_thr(Mtup, 12)
            fixtup10 = rp.ma_thr(Mtup, 4)
            fixtup0 = rp.ma_thr(Mtup, 0)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup30 = rp.ma_thr(Mtup, 12)
            fixtup10 = rp.ma_thr(Mtup, 4)
            fixtup0 = rp.ma_thr(Mtup, 0)
        Mtups30.append(fixtup30)
        Mtups10.append(fixtup10)
        Mtups0.append(fixtup0)
    for tupList in Mtups30:
        for x in tupList:
            if x[0]=='R':
                rems30.append(x[1])
            elif x[0]=='W':
                wakes30.append(x[1])
            else:
                nrems30.append(x[1])
    for tupList in Mtups10:
        for x in tupList:
            if x[0]=='R':
                rems10.append(x[1])
            elif x[0]=='W':
                wakes10.append(x[1])
            else:
                nrems10.append(x[1])
    for tupList in Mtups0:
        for x in tupList:
            if x[0]=='R':
                rems0.append(x[1])
            elif x[0]=='W':
                wakes0.append(x[1])
            else:
                nrems0.append(x[1])                
    print(rec)

stats30 = [sum(rems30), sum(nrems30), sum(wakes30)]
stats10 = [sum(rems10), sum(nrems10), sum(wakes10)]
stats0 = [sum(rems0), sum(nrems0), sum(wakes0)]
        
lbls = ['REM', '|N|', '|W|']
cols = ['cyan', 'gray', 'purple']

fig1a = plt.figure()
plt.title('MA = 30 s')
plt.pie(stats30, labels=lbls, autopct='%1.1f%%', colors=cols)

fig1b = plt.figure()
plt.title('MA = 10 s')
plt.pie(stats10, labels=lbls, autopct='%1.1f%%', colors=cols)

fig1c = plt.figure()
plt.title('No MA')
plt.pie(stats0, labels=lbls, autopct='%1.1f%%', colors=cols)      
            

fig1a.savefig(outpath + 'pie30.pdf')
fig1b.savefig(outpath + 'pie10.pdf')
fig1c.savefig(outpath + 'pie0.pdf')



###############################################################################
from sklearn.mixture import GaussianMixture

# Model with MA = 10 s

rem30DF = rp.standard_recToDF(ppath, recordings, 12, False)
rem10DF = rp.standard_recToDF(ppath, recordings, 4, False)
rem0DF = rp.standard_recToDF(ppath, recordings, 0, False)


split10DF = rp.splitData(rem10DF, 30)
split10logsws = [x.logsws for x in split10DF]

#Set seed
np.random.seed(1)

#Lists to store parameters
klow10 = []
khigh10 = []
mlow10 = []
mhigh10 = []
slow10 = []
shigh10 = []

#Go through each 30s split and estimate GMM parameters
for idx,data in enumerate(split10logsws):
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
            klow10.append(k1)
            khigh10.append(k2)
            mlow10.append(m1)
            mhigh10.append(m2)
            slow10.append(s1)
            shigh10.append(s2)
        else:
            klow10.append(k2)
            khigh10.append(k1)
            mlow10.append(m2)
            mhigh10.append(m1)
            slow10.append(s2)
            shigh10.append(s1)
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
        khigh10.append(k)
        mhigh10.append(m)
        shigh10.append(s)
        klow10.append(None)
        mlow10.append(None)
        slow10.append(None)

xes = np.arange(15,240,30)


# MA = 0
split0DF = rp.splitData(rem0DF, 30)
split0logsws = [x.logsws for x in split0DF]

#Set seed
np.random.seed(1)

#Lists to store parameters
klow0 = []
khigh0 = []
mlow0 = []
mhigh0 = []
slow0 = []
shigh0 = []

#Go through each 30s split and estimate GMM parameters
for idx,data in enumerate(split0logsws):
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
            klow0.append(k1)
            khigh0.append(k2)
            mlow0.append(m1)
            mhigh0.append(m2)
            slow0.append(s1)
            shigh0.append(s2)
        else:
            klow0.append(k2)
            khigh0.append(k1)
            mlow0.append(m2)
            mhigh0.append(m1)
            slow0.append(s2)
            shigh0.append(s1)
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
        khigh0.append(k)
        mhigh0.append(m)
        shigh0.append(s)
        klow0.append(None)
        mlow0.append(None)
        slow0.append(None)


# MA = 30 s
split30DF = rp.splitData(rem30DF, 30)
split30logsws = [x.logsws for x in split30DF]

#Set seed
np.random.seed(1)

#Lists to store parameters
klow30 = []
khigh30 = []
mlow30 = []
mhigh30 = []
slow30 = []
shigh30 = []

#Go through each 30s split and estimate GMM parameters
for idx,data in enumerate(split30logsws):
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
            klow30.append(k1)
            khigh30.append(k2)
            mlow30.append(m1)
            mhigh30.append(m2)
            slow30.append(s1)
            shigh30.append(s2)
        else:
            klow30.append(k2)
            khigh30.append(k1)
            mlow30.append(m2)
            mhigh30.append(m1)
            slow30.append(s2)
            shigh30.append(s1)
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
        khigh30.append(k)
        mhigh30.append(m)
        shigh30.append(s)
        klow30.append(None)
        mlow30.append(None)
        slow30.append(None)




xes = np.arange(15,240,30)

plt.figure()
plt.subplot(2,3,1)
plt.scatter(xes, khigh, s=20, color='k', marker="1", label="MA=20")
plt.scatter(xes, khigh10, s=20, color='k', marker="x", label="MA=10")
plt.scatter(xes, khigh0, s=20, color='k', marker="s", label="MA=0")
plt.xlim([0,240])
sns.despine()
plt.legend()
plt.subplot(2,3,2)
plt.scatter(xes, mhigh, s=10, color=col1, marker="1")
plt.scatter(xes, mhigh10, s=10, color=col1, marker="x")
plt.scatter(xes, mhigh0, s=10, color=col1, marker="s")
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,3)
plt.scatter(xes, shigh, s=10, color=col1, marker="1")
plt.scatter(xes, shigh10, s=10, color=col1, marker="x")
plt.scatter(xes, shigh0, s=10, color=col1, marker="s")
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,5)
plt.scatter(xes, mlow, s=10, color=col2, marker="1")
plt.scatter(xes, mlow10, s=10, color=col2, marker="x")
plt.scatter(xes, mlow0, s=10, color=col2, marker="s")
plt.xlim([0,240])
sns.despine()
plt.subplot(2,3,6)
plt.scatter(xes, slow, s=10, color=col2, marker="1")
plt.scatter(xes, slow10, s=10, color=col2, marker="x")
plt.scatter(xes, slow0, s=10, color=col2, marker="s")
plt.xlim([0,240])
sns.despine()


###############################################################################

###### LF-test for MA = 10 s

ks_sims10 = []
ks_obs10 = []

from scipy.stats import rv_continuous
xvals = np.arange(0,10,0.01)

for idx,subDF in enumerate(split10DF):
    kssim = []
    subdata = list(subDF.logsws)
    k1 = khigh10[idx]
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh10[idx]
    s1 = shigh10[idx]
    m2 = mlow10[idx]
    s2 = slow10[idx]
    
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
    ks_sims10.append(kssim)
    ks_obs10.append(ksobs)

teststats10 = [x[0] for x in ks_obs10]

pvals10 = []
for idx,kssim in enumerate(ks_sims10):
    ksobs = teststats10[idx]
    pval = (sum(kssim>ksobs))/(10000)
    pvals10.append(pval)

    
    
###### LF-test for MA = 0 s

ks_sims0 = []
ks_obs0 = []

from scipy.stats import rv_continuous
xvals = np.arange(0,10,0.01)

for idx,subDF in enumerate(split0DF):
    kssim = []
    subdata = list(subDF.logsws)
    k1 = khigh0[idx]
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh0[idx]
    s1 = shigh0[idx]
    m2 = mlow0[idx]
    s2 = slow0[idx]
    
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
    ks_sims0.append(kssim)
    ks_obs0.append(ksobs)

teststats0 = [x[0] for x in ks_obs0]

pvals0 = []
for idx,kssim in enumerate(ks_sims0):
    ksobs = teststats0[idx]
    pval = (sum(kssim>ksobs))/(10000)
    pvals0.append(pval)



###### LF-test for MA = 30 s

ks_sims30 = []
ks_obs30 = []

from scipy.stats import rv_continuous
xvals = np.arange(0,10,0.01)

for idx,subDF in enumerate(split30DF):
    kssim = []
    subdata = list(subDF.logsws)
    k1 = khigh30[idx]
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh30[idx]
    s1 = shigh30[idx]
    m2 = mlow30[idx]
    s2 = slow30[idx]
    
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
    ks_sims30.append(kssim)
    ks_obs30.append(ksobs)

teststats30 = [x[0] for x in ks_obs30]

pvals30 = []
for idx,kssim in enumerate(ks_sims30):
    ksobs = teststats30[idx]
    pval = (sum(kssim>ksobs))/(10000)
    pvals30.append(pval)


#############################

gmm30DF = pd.DataFrame(list(zip(khigh30,mhigh30,shigh30,mlow30,slow30)), columns = ['khigh','mhigh','shigh','mlow','slow'])
gmm10DF = pd.DataFrame(list(zip(khigh10,mhigh10,shigh10,mlow10,slow10)), columns = ['khigh','mhigh','shigh','mlow','slow'])
gmm0DF = pd.DataFrame(list(zip(khigh0,mhigh0,shigh0,mlow0,slow0)), columns = ['khigh','mhigh','shigh','mlow','slow'])

gmm30DF.to_csv('gmm30DF.csv')
gmm10DF.to_csv('gmm10DF.csv')
gmm0DF.to_csv('gmm0DF.csv')