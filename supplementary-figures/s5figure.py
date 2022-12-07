#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 19:30:04 2021

@author: cwlab08
"""
##### Preparations

#Import necessary packages
import os
import rempropensity as rp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/REM_GMM/final_figures/Sfig2/'

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

#GMM parameters
klow = gmmDF.klow
khigh = gmmDF.khigh
mlow = gmmDF.mlow
mhigh = gmmDF.mhigh
slow = gmmDF.slow
shigh = gmmDF.shigh

#Log-model coefficients
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

#Linear=model coefficients
klow_lina = linfitDF.klow[0]
klow_linb = linfitDF.klow[1]
khigh_lina = linfitDF.khigh[0]
khigh_linb = linfitDF.khigh[1]
mlow_lina = linfitDF.mlow[0]
mlow_linb = linfitDF.mlow[1]
mhigh_lina = linfitDF.mhigh[0]
mhigh_linb = linfitDF.mhigh[1]
slow_lina = linfitDF.slow[0]
slow_linb = linfitDF.slow[1]
shigh_lina = linfitDF.shigh[0]
shigh_linb = linfitDF.shigh[1]

###############################################################################
### SFig1A - Model fit -- linear log comparison

#Log function
def log_func(x,a,b,c):
    return a*np.log(x+b)+c

#Linear function
def lin_func(x,a,b):
    return a*x+b

#x-values of plot
xes = np.arange(15,240,30)
xplot = np.arange(0,240,0.1)

#Plot gmm parameters with both linear and log fits
sfig1A1 = plt.figure()
plt.scatter(xes[0:6], khigh[0:6], s=10, color='k')
plt.scatter(xes[6:], khigh[6:], s=10, facecolors='white', edgecolors='k')
plt.plot(xplot,khigh_loga*np.log(xplot+khigh_logb)+khigh_logc, color='k')
plt.plot(xplot,khigh_lina*xplot+khigh_linb, color='k', ls='--')
plt.xticks([0,100,200])
plt.xlim([0,240])
plt.ylim([0,1.1])
sns.despine()

sfig1A2 = plt.figure()
plt.scatter(xes, mhigh, s=10, color=col1)
plt.plot(xplot,mhigh_loga*np.log(xplot+mhigh_logb)+mhigh_logc, color=col1)
plt.plot(xplot,mhigh_lina*xplot+mhigh_linb, color=col1, ls='--')
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()

sfig1A3 = plt.figure()
plt.scatter(xes, shigh, s=10, color=col1)
plt.plot(xplot,shigh_loga*np.log(xplot+shigh_logb)+shigh_logc, color=col1)
plt.plot(xplot,shigh_lina*xplot+shigh_linb, color=col1, ls='--')
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()

sfig1A4 = plt.figure()
plt.scatter(xes, mlow, s=10, color=col2)
plt.plot(xplot, mlow_loga*np.log(xplot+mlow_logb)+mlow_logc, color=col2)
plt.plot(xplot, mlow_lina*xplot+mlow_linb, color=col2, ls='--')
plt.xlim([0,240])
plt.ylim([3,6])
plt.xticks([0,100,200])
sns.despine()

sfig1A5 = plt.figure()
plt.scatter(xes, slow, s=10, color=col2)
plt.plot(xplot, slow_loga*np.log(xplot+slow_logb)+slow_logc, color=col2)
plt.plot(xplot, slow_lina*xplot+slow_linb, color=col2, ls='--')
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()

sfig1A1.savefig(outpath+'sf1A_khigh.pdf')
sfig1A2.savefig(outpath+'sf1A_mhigh.pdf')
sfig1A3.savefig(outpath+'sf1A_shigh.pdf')
sfig1A4.savefig(outpath+'sf1A_mlow.pdf')
sfig1A5.savefig(outpath+'sf1A_slow.pdf')

##Calculate RSS for both log and linear fits
#khigh
logres1 = []
linres1 = []

for idx,x in enumerate(np.arange(15,240,30)[0:6]):
    logy = log_func(x,khigh_loga,khigh_logb,khigh_logc)
    liny = lin_func(x,khigh_lina,khigh_linb)
    y = khigh[idx]
    logres1.append(y-logy)
    linres1.append(y-liny)

rsslog1 = np.sum(np.square(logres1))
rsslin1 = np.sum(np.square(linres1))

#mhigh
logres2 = []
linres2 = []

for idx,x in enumerate(np.arange(15,240,30)):
    logy = log_func(x,mhigh_loga,mhigh_logb,mhigh_logc)
    liny = lin_func(x,mhigh_lina,mhigh_linb)
    y = mhigh[idx]
    logres2.append(y-logy)
    linres2.append(y-liny)    

rsslog2 = np.sum(np.square(logres2))
rsslin2 = np.sum(np.square(linres2))

#shigh
logres3 = []
linres3 = []

for idx,x in enumerate(np.arange(15,240,30)):
    logy = log_func(x,shigh_loga,shigh_logb,shigh_logc)
    liny = lin_func(x,shigh_lina,shigh_linb)
    y = shigh[idx]
    logres3.append(y-logy)
    linres3.append(y-liny)    

rsslog3 = np.sum(np.square(logres3))
rsslin3 = np.sum(np.square(linres3))

#mlow
logres4 = []
linres4 = []

for idx,x in enumerate(np.arange(15,210,30)[0:5]):
    logy = log_func(x,mlow_loga,mlow_logb,mlow_logc)
    liny = lin_func(x,mlow_lina,mlow_linb)
    y = mlow[idx]
    logres4.append(y-logy)
    linres4.append(y-liny)

rsslog4 = np.sum(np.square(logres4))
rsslin4 = np.sum(np.square(linres4))

#slow
logres5 = []
linres5 = []

for idx,x in enumerate(np.arange(15,210,30)[0:5]):
    logy = log_func(x,slow_loga,slow_logb,slow_logc)
    liny = lin_func(x,slow_lina,slow_linb)
    y = slow[idx]
    logres5.append(y-logy)
    linres5.append(y-liny)

rsslog5 = np.sum(np.square(logres5))
rsslin5 = np.sum(np.square(linres5))

###############################################################################
###Sfig1B - Model boundary

#x-values for plot
xplot = np.arange(2,9,0.1)

#Plot pdfs for low and high gaussian for REMpre in [2.5,12.5]
sfig1B = plt.figure(figsize=(12,4))
plt.subplots_adjust(wspace=0.35)
for idx,rem in enumerate(np.arange(2.5,15,2.5)):
    plt.subplot(1,5,idx+1)
    plt.title('REM='+str(rem))
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    s2 = slow_lina*rem+slow_linb
    f1 = lambda x: k1*stats.norm.pdf(x,m1,s1)
    f2 = lambda x: k2*stats.norm.pdf(x,m2,s2)
    plt.plot(xplot, f1(xplot), color=col1)
    plt.plot(xplot, f2(xplot), color=col2)
    plt.ylim([0,0.41])
    plt.yticks([0,0.1,0.2,0.3,0.4])
    sns.despine()

sfig1B.savefig(outpath+'sf1B_modelboundary.pdf')

###############################################################################
###Sfig1C - Model simulation

#List to store simulation results
simres = []

#subset of original dataset s.t. REMpre is in [7.5,240]
subDF = remDF.loc[(remDF.rem>=7.5)&(remDF.rem<240)]

#Using REMpre of original dataset and GMM, simulate 10,000 iterations of |N|
for i in range(10000):
    print(i)
    for rem in subDF.rem:
        up = True
        threshold = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
        if threshold>1:
            threshold=1
        randnpick = np.random.uniform(0,1)
        if randnpick > threshold:
            up=False
        
        if up:
            mean = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
            std = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
        else:
            mean = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
            std = slow_lina*rem+slow_linb
        
        logsws = np.random.normal(mean, std)
        sws = np.exp(logsws)
        simres.append(sws)

#Simulation results in log-scale
logsimres = np.log(simres)

#Define histogram bins of log and normal scale
y1,bbins1 = np.histogram(subDF.sws, bins=30)
y2,bbins2 = np.histogram(subDF.logsws, bins=30)

#Plot histogram of simulation and data on normal scale
sfig2C1 = plt.figure()
plt.hist(simres, bins=bbins1, color='red', alpha=0.5, label='simulation', density=1)
plt.hist(subDF.sws, bins=bbins1, color='blue', alpha=0.5, label='data', density=1)
sns.despine()
plt.legend()

#Plot histogram of simulation and data on log scale
sfig2C2 = plt.figure()
plt.hist(logsimres, bins=bbins2, color='red', alpha=0.5, label='simulation', density=1)
plt.hist(subDF.logsws, bins=bbins2, color='blue', alpha=0.5, label='data', density=1)
sns.despine()
plt.legend()


sfig2C1.savefig(outpath+'sf2C_simres.pdf')
sfig2C2.savefig(outpath+'sf2C_logsimres.pdf')