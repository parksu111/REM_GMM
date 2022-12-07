#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:30:18 2021

@author: cwlab08
"""

import os
import rempropensity as rp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

#Set path for figures
outpath = r'/home/cwlab08/Desktop/REM_GMM/final_figures/Sfig7/'

gmmDF = pd.read_csv('darkgmmDF.csv')
linfitDF = pd.read_csv('darklinfitDF.csv')
logfitDF = pd.read_csv('darklogfitDF.csv')

khigh = gmmDF.khigh
mlow = gmmDF.mlow
mhigh = gmmDF.mhigh
slow = gmmDF.slow
shigh = gmmDF.shigh

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
#### SFig6 A - Compare linear and log fits for gmm parameters

#Exclude 'none' from lists for low distribution parameters
amlow = [mlow[0],mlow[1],mlow[2],mlow[5]]
aslow = [slow[0],slow[1],slow[2],slow[5]]

#Define log or linear functions
def log_func(x,a,b,c):
    return a*np.log(x+b)+c

def lin_func(x,a,b):
    return a*x+b

#xvalues to plot
xes = np.arange(15,240,30)
lowxes = [15,45,75,165]
xplot1 = np.arange(0,240)
xplot2 = np.arange(0,180)

#Plot
sfig6A1 = plt.figure(figsize=(5,5))
plt.title('khigh')
plt.scatter(xes, khigh, s=10, color='darkslateblue')
plt.plot(xplot1, log_func(xplot1,khigh_loga,khigh_logb,khigh_logc),color='darkslateblue')
plt.plot(xplot1, lin_func(xplot1, khigh_lina, khigh_linb),color='darkslateblue', ls='--')
sns.despine()
plt.ylim([0,1.4])
plt.xlim([0,240])

sfig6A2 = plt.figure(figsize=(5,5))
plt.title('mhigh')
plt.scatter(xes, mhigh, s=10,color='darkslateblue')
plt.plot(xplot1, log_func(xplot1,mhigh_loga,mhigh_logb,mhigh_logc),color='darkslateblue')
plt.plot(xplot1, lin_func(xplot1,mhigh_lina,mhigh_linb),color='darkslateblue', ls='--')
sns.despine()
plt.xlim([0,240])

sfig6A3 = plt.figure(figsize=(5,5))
plt.title('shigh')
plt.scatter(xes, shigh, s=10, color='darkslateblue')
plt.plot(xplot1, log_func(xplot1,shigh_loga,shigh_logb,shigh_logc),color='darkslateblue')
plt.plot(xplot1, lin_func(xplot1,shigh_lina,shigh_linb),color='darkslateblue', ls='--')
sns.despine()
plt.xlim([0,240])

sfig6A4 = plt.figure(figsize=(5,5))
plt.title('mlow')
plt.scatter(lowxes, amlow, s=10, color='darkslateblue')
plt.plot(xplot2, log_func(xplot2, mlow_loga,mlow_logb,mlow_logc),color='darkslateblue')
plt.plot(xplot2, lin_func(xplot2, mlow_lina,mlow_linb),color='darkslateblue', ls='--')
sns.despine()
plt.xlim([0,240])
plt.ylim([2,7])

sfig6A5 = plt.figure(figsize=(5,5))
plt.title('slow')
plt.scatter(lowxes, aslow, s=10, color='darkslateblue')
plt.plot(xplot2, log_func(xplot2, slow_loga,slow_logb,slow_logc),color='darkslateblue')
plt.plot(xplot2, lin_func(xplot2, slow_lina,slow_linb),color='darkslateblue', ls='--')
sns.despine()
plt.xlim([0,240])

#Save figures
sfig6A1.savefig(outpath+'sf6a_khigh.pdf')
sfig6A2.savefig(outpath+'sf6a_mhigh.pdf')
sfig6A3.savefig(outpath+'sf6a_shigh.pdf')
sfig6A4.savefig(outpath+'sf6a_mlow.pdf')
sfig6A5.savefig(outpath+'sf6a_slow.pdf')


#### Calculate RSS for linear and log fits

#khigh

logres1 = []
linres1 = []

for idx,x in enumerate(np.arange(15,240,30)[0:4]):
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

for idx,x in enumerate(lowxes):
    logy = log_func(x,mlow_loga,mlow_logb,mlow_logc)
    liny = lin_func(x,mlow_lina,mlow_linb)
    y = amlow[idx]
    logres4.append(y-logy)
    linres4.append(y-liny)

rsslog4 = np.sum(np.square(logres4))
rsslin4 = np.sum(np.square(linres4))


#slow
logres5 = []
linres5 = []

for idx,x in enumerate(lowxes):
    logy = log_func(x,slow_loga,slow_logb,slow_logc)
    liny = lin_func(x,slow_lina,slow_linb)
    y = aslow[idx]
    logres5.append(y-logy)
    linres5.append(y-liny)

rsslog5 = np.sum(np.square(logres5))
rsslin5 = np.sum(np.square(linres5))

###############################################################################
#### Sfig6B - Plot low and high distributions for REMpre in [5,15]

#Define colors
current_palette = sns.color_palette('muted', 10)
col1 = current_palette[3]
col2 = current_palette[0]

#xvalues to plot
xplot = np.arange(2,9,0.1)

#Plot
sfig6B = plt.figure(figsize=(14,4))
plt.subplots_adjust(wspace=0.35)
for idx,rem in enumerate(np.arange(5,17.5,2.5)):
    plt.subplot(1,5,idx+1)
    plt.title(rem)
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    s2 = slow_lina*rem+slow_linb
    f1 = lambda x: k1*stats.norm.pdf(x,m1,s1)
    f2 = lambda x: k2*stats.norm.pdf(x,m2,s2)
    plt.plot(xplot, f1(xplot), color=col1, lw=1)
    plt.plot(xplot, f2(xplot), color=col2, lw=1)
    sns.despine()
    plt.ylim([0,0.35])
    plt.yticks([0,0.1,0.2,0.3])

sfig6B.savefig(outpath+'sf6b_lowhighboundary.pdf')