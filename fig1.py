#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:04:11 2020

@author: cwlab08
"""
##### Preparations

#Import necessary packages
import os
import rempropensity as rp
import statsmodels.api as sm
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path to output figures
outpath = r'/home/cwlab08/Desktop/final_figures/1_Def/'

#Make dataframe containing all REM-NREM cycles
remDF = rp.standard_recToDF(ppath, recordings, 8, False)


###############################################################################

#Fig1B - Scatter plots comparing REMpre vs. |N|, inter-REM, |W|

#inter-REM
x1 = remDF.rem
x1 = sm.add_constant(x1)
y1 = remDF.inter
mod1 = sm.OLS(y1,x1)
res1 = mod1.fit()
linreg1 = lambda x: res1.params[1]*x + res1.params[0]

xplot = np.arange(0,max(remDF.rem),1)

fig2B1 = plt.figure()
plt.scatter(remDF.rem, remDF.inter, s=10, color='gray', alpha=0.5)
plt.plot(xplot, linreg1(xplot), color='r', ls='--')
sns.despine()
plt.xticks([0,60,120,180,240,300],["","","","","",""])
plt.yticks([0,2000,4000,6000,8000],["","","","",""])

#|N|
x2 = remDF.rem
x2 = sm.add_constant(x2)
y2 = remDF.sws
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()
linreg2 = lambda x: res2.params[1]*x + res2.params[0]

fig2B2 = plt.figure()
plt.scatter(remDF.rem, remDF.sws, s=10, color='gray', alpha=0.5)
plt.plot(xplot, linreg2(xplot), color='r', ls='--')
sns.despine()
plt.xticks([0,60,120,180,240,300],["","","","","",""])
plt.yticks([0,1000,2000,3000],["","","",""])


#|W|
x3 = remDF.rem
x3 = sm.add_constant(x3)
y3 = remDF.wake
mod3 = sm.OLS(y3,x3)
res3 = mod3.fit()
linreg3 = lambda x: res3.params[1]*x +res3.params[0]

fig2B3 = plt.figure()
plt.scatter(remDF.rem, remDF.wake, s=10, color='gray', alpha=0.5)
plt.plot(xplot, linreg3(xplot), color='r', ls='--')
sns.despine()
plt.xticks([0,60,120,180,240,300],["","","","","",""])
plt.yticks([0,2000,4000,6000],["","","",""])

fig2B1.savefig(outpath+'f2B1_reminterscatter.png')
fig2B2.savefig(outpath+'f2B2_remnremscatter.png')
fig2B3.savefig(outpath+'f2B3_remwakescatter.png')

res1.rsquared
res2.rsquared
res3.rsquared
