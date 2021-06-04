    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:11:14 2020

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
outpath = r'/home/cwlab08/Desktop/final_figures/2_GMM/'

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
###############################################################################
###Figure 2A - Log SWS Scatter

#Split Data by 30s of REMpre
splitDFs = rp.splitData(remDF,30)

#Calculate mean and standard deviation of ln(|N|)
logmeans = [np.mean(x.logsws) for x in splitDFs]
logstds = [np.std(x.logsws) for x in splitDFs]


xes = np.arange(15,240,30)

fig2A = plt.figure()
plt.scatter(remDF.rem, remDF.logsws, s=10, alpha=0.5, color='gray')
plt.plot(xes,logmeans, color='k')
plt.errorbar(xes,logmeans,logstds, color='k')
for xv in np.arange(30,240,30):
    plt.axvline(x=xv, color='k', ls='--',lw=0.5)
plt.xticks([0,60,120,180],["","","",""])
plt.yticks([2,4,6,8],["","","",""])
sns.despine()

fig2A.savefig(outpath+'f2A_logscatter.png')


###############################################################################
###Figure 2B - Log SWS histogram split by 30s of REMpre

#Split Data by 30s of REMpre
splitDFs = rp.splitData(remDF,30)
splitLogSws = [x.logsws for x in splitDFs]

#x-values of plot
xplot = np.arange(2,9,0.1)

#Bins of histogram
y1,bbins1 = np.histogram(remDF.logsws, bins=30)

#Calulate and plot mixture pdf along with histogram
fig2B = plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for idx,data in enumerate(splitLogSws):
    k1 = khigh[idx]
    if k1>1:
        k1=1
    k2 = klow[idx]
    m1 = mhigh[idx]
    m2 = mlow[idx]
    s1 = shigh[idx]
    s2 = slow[idx]
    if idx<5:
        f = lambda x: k1*stats.norm.pdf(x,m1,s1) + k2*stats.norm.pdf(x,m2,s2)
    else:
        f = lambda x: k1*stats.norm.pdf(x,m1,s1)
    plt.subplot(2,4,idx+1)
    plt.hist(data, bins=bbins1, color='gray', alpha=0.6, density=1)
    plt.plot(xplot, f(xplot), color='k', ls='--', lw=1)
    sns.despine()
    if idx<4:
        plt.ylim([0,1.0])
        plt.yticks([0,0.5,1.0])
    elif idx<7:
        plt.ylim([0,1.4])
        plt.yticks([0,0.5,1.0])
    plt.xticks([2,4,6,8])
    
fig2B.savefig(outpath+'f2B_mixpdf.pdf')

###############################################################################
###Figure 2C - Example GMM; EM algorithm

#Example split is 30<=REMpre<60
exdata = splitLogSws[1]

#GMM parameters
k1 = khigh[1]
k2 = klow[1]
m1 = mhigh[1]
m2 = mlow[1]
s1 = shigh[1]
s2 = slow[1]

#GMM function
f1 = lambda x: k1*stats.norm.pdf(x,m1,s1)
f2 = lambda x: k2*stats.norm.pdf(x,m2,s2)
fmix = lambda x: k1*stats.norm.pdf(x,m1,s1) + k2*stats.norm.pdf(x,m2,s2)

fig2C = plt.figure()
plt.hist(exdata, bins=bbins1, density=1, color='gray', alpha=0.6)
plt.plot(xplot, f1(xplot), color=col1)
plt.plot(xplot, f2(xplot), color=col2)
plt.plot(xplot, fmix(xplot), color='k', ls='--', lw=1)
plt.xticks([2,4,6,8])
plt.yticks([0,0.2,0.4,0.6])
sns.despine()

fig2C.savefig(outpath+'f2C_gmmexample.pdf')

###############################################################################
###Figure 2D - GMM Parameters

#intersection of khigh and 1
tfun = lambda x: khigh_loga*np.log(x+khigh_logb)+khigh_logc - 1
kint = fsolve(tfun, x0=[160])

#x-values of plot
xes = np.arange(15,240,30)
xplot = np.arange(0,240,0.1)
xplot1a = np.arange(0,kint,0.1)

#Plot parameters along with log or linear fits
fig2D = plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.subplot(2,3,1)
plt.scatter(xes, khigh, s=10, color='k')
plt.plot(xplot1a,khigh_loga*np.log(xplot1a+khigh_logb)+khigh_logc, color='k')
plt.axhline(y=1, color='k')
plt.xticks([0,100,200])
plt.xlim([0,240])
plt.ylim([0,1.1])
sns.despine()
plt.subplot(2,3,2)
plt.scatter(xes, mhigh, s=10, color=col1)
plt.plot(xplot,mhigh_loga*np.log(xplot+mhigh_logb)+mhigh_logc, color=col1)
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()
plt.subplot(2,3,3)
plt.scatter(xes, shigh, s=10, color=col1)
plt.plot(xplot,shigh_loga*np.log(xplot+shigh_logb)+shigh_logc, color=col1)
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()
plt.subplot(2,3,5)
plt.scatter(xes, mlow, s=10, color=col2)
plt.plot(xplot, mlow_loga*np.log(xplot+mlow_logb)+mlow_logc, color=col2)
plt.xlim([0,240])
plt.ylim([3,6])
plt.xticks([0,100,200])
sns.despine()
plt.subplot(2,3,6)
plt.scatter(xes, slow, s=10, color=col2)
plt.plot(xplot, slow_lina*xplot+slow_linb, color=col2)
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()

fig2D.savefig(outpath+'f2D_gmmparams.pdf')

###############################################################################
###Figure 2E - PDF Heatmap

result = []

#For each 5s increment of REMpre in [2.5,247.5], calculate the integrals of 
#the conditional PDF for 50 s splits of |N| 
for rem in np.arange(10,250,5):
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    s2 = slow_lina*rem+slow_linb
    if k2==0:
        f = lambda x,a,b,k: k*stats.lognorm.pdf(x,b,0,np.exp(a))
    else:
        f = lambda x,a1,b1,k1,a2,b2,k2: k1*stats.lognorm.pdf(x,b1,0,np.exp(a1))+k2*stats.lognorm.pdf(x,b2,0,np.exp(a2))
    
    subres = []
    for y in np.arange(25,2500,50):
        lowbound = y-25
        highbound = y+25
        if k2==0:
            midres = quad(f,lowbound,highbound, args=(m1,s1,k1))[0]
        else:
            midres = quad(f,lowbound,highbound, args=(m1,s1,k1,m2,s2,k2))[0]
        if rem<7.5:
            subres.append(0)
        else:
            subres.append(midres)
    print(rem)
    result.append(subres)

#Change result into array and transpose
resarray = np.array(result)
resarray2 = resarray.transpose()

fig2E = plt.figure()
plt.imshow(resarray2, cmap='hot',interpolation='nearest',origin='lower')
plt.xticks(ticks=[0,10,22,34,46],labels=['10','60','120','180','240'])
plt.yticks(ticks=[0,9,19,29,39],labels=['50','500','1000','1500','2000'])
plt.colorbar()
sns.despine()

fig2E.savefig(outpath+'f2E_pdfheatmap.pdf')


###############################################################################
###Figure 2F - Plot CDFs of gmm for various values of REMpre

#x-values of plot
xplot = np.arange(0,2500,1)

#CDF values at |N|=2000s
cdf2000s = []

#Calculate CDF of GMM for values of REMpre and plot
fig2F = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=15)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,rem in enumerate(np.arange(30,240,30)):
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    s2 = slow_lina*rem+slow_linb
    
    tfun = lambda x: k1*stats.lognorm.cdf(x,s1,0,np.exp(m1))+k2*stats.lognorm.cdf(x,s2,0,np.exp(m2))
    colorVal = scalarMap.to_rgba(2*idx+1)
    lbl = '$REM_{pre}$='+str(rem)
    plt.plot(xplot, tfun(xplot), label=lbl, color=colorVal)
    cdf2000s.append(tfun(2000))
plt.legend()
plt.xlim([0,2500])
plt.xticks([0,1000,2000])
sns.despine()

fig2F.savefig(outpath+'f2F_allcdfs.pdf')

#Check minimum cdf value at |N|=2000s
min(cdf2000s)
