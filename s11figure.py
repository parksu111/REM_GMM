#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:49:59 2021

@author: cwlab08
"""
#Import necessary packages
import os
import rempropensity as rp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.optimize import fsolve
from scipy import stats
import statsmodels.api as sm

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/final_figures/Sfig5/'

#Define colors
current_palette = sns.color_palette('muted', 10)
col1 = current_palette[3]
col2 = current_palette[0]
palette2 = sns.color_palette('pastel',10)
colwo = palette2[7]
colww = palette2[4]

#Make dataframe containing all NREM-REM cycles
remDF = rp.standard_recToDF(ppath, recordings, 8, False)
remDF = remDF.loc[remDF.rem<240]
nonDF = remDF.loc[remDF.rem<7.5]
subDF = remDF.loc[remDF.rem>=7.5]
subDF = subDF.reset_index(drop=True)

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

khigh_loga = logfitDF.khigh[0]
khigh_logb = logfitDF.khigh[1]
khigh_logc = logfitDF.khigh[2]
mlow_loga = logfitDF.mlow[0]
mlow_logb = logfitDF.mlow[1]
mlow_logc = logfitDF.mlow[2]
mhigh_loga = logfitDF.mhigh[0]
mhigh_logb = logfitDF.mhigh[1]
mhigh_logc = logfitDF.mhigh[2]
shigh_loga = logfitDF.shigh[0]
shigh_logb = logfitDF.shigh[1]
shigh_logc = logfitDF.shigh[2]
slow_lina = linfitDF.slow[0]
slow_linb = linfitDF.slow[1]


# Find intersection between low and high gaussian distributions
intersection = []

for rem in np.arange(7.5,242.5,2.5):
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    s2 = slow_lina*rem+slow_linb
    if k2>0:
        tfun = lambda x: k1*stats.norm.pdf(x,m1,s1)-k2*stats.norm.pdf(x,m2,s2)
        intersection.append((rem, fsolve(tfun, x0=[5.25], maxfev=999999)))

intersect_x = np.array([x[0] for x in intersection])
intersect_y = np.array([x[1] for x in intersection])
exp_intersect_y = np.exp(intersect_y)


#Divide Dataset into Sequential and Single Cycles
seqrows = []
sinrows = []

for index,row in subDF.iterrows():
    rem = row['rem']
    sws = row['sws']
    if rem > max(intersect_x):
        sinrows.append(row)
    else:
        indBurst = np.where(intersect_x==rem)[0][0]
        burstLim = exp_intersect_y[indBurst][0]
        if sws<burstLim:
            seqrows.append(row)
        else:
            sinrows.append(row)

seqDF = pd.DataFrame(seqrows)
sinDF = pd.DataFrame(sinrows)


#Find 1st percntile of high distribution
logthresh1 = []

for rem in np.arange(7.5,242.5,2.5):
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    logthresh1.append(m1-2.326*s1)

thresh1 = np.exp(logthresh1)


###############################################################################
####SFig4 A,B - |N| by |W| and #wake blocks

#lists to store results
wakecounts = []
totwakes = []
rems = []
swss = []
wakeblocks = []

#Loop through recordings and for each single cycle, calculate |W|, |N|, #wake blocks
for rec in recordings: 
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    Mtups = []
    for idx,x in enumerate(recs):
        Mvec = rp.nts(x)
        if idx==0:
            Mtup = rp.vecToTup(Mvec, start=0)
            fixtup = rp.ma_thr(Mtup, 8)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 8)
        Mtups.append(fixtup)
    for tupList in Mtups:
        tupList2 = tupList[1:len(tupList)-1]
        nrt_locs = rp.nrt_locations(tupList2)
        cnt2 = 0
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            for y in sub:
                if (y[0]=='N')or(y[0]=='MA'):
                    sws+=y[1]*2.5
            if (rem>=7.5)&(rem<240):
                if rp.isSequential(rem,sws, intersect_x,intersect_y)==False:            
                    wakecnt = 0
                    totwake = 0
                    for x in sub:
                        if x[0]=='W':
                            wakecnt+=1
                            totwake+=x[1]*2.5
                            wakeblocks.append(x[1]*2.5)
                    wakecounts.append(wakecnt)
                    totwakes.append(totwake)
                    rems.append(rem)
                    swss.append(sws)
            cnt2+=1
    print(rec)

#
whDF = pd.DataFrame(list(zip(rems,swss,wakecounts,totwakes)),columns=['rem','sws','wcount','twake'])
wakeDF = pd.DataFrame(list(zip(rems,swss,wakecounts,totwakes)),columns=['rem','sws','wcount','twake'])
ww_wakeDF = wakeDF.loc[wakeDF.twake>0]
wo_wakeDF = wakeDF.loc[wakeDF.twake==0]

#Split data by # wake blocks
whDF0 = whDF.loc[whDF.wcount==0]
whDF1 = whDF.loc[whDF.wcount==1]
whDF2 = whDF.loc[whDF.wcount==2]
whDF3 = whDF.loc[whDF.wcount==3]
whDF4 = whDF.loc[whDF.wcount==4]
whDF5 = whDF.loc[whDF.wcount==5]
whDF6 = whDF.loc[whDF.wcount==6]
whDF7 = whDF.loc[whDF.wcount==7]
whDF8 = whDF.loc[whDF.wcount==8]
whDF9 = whDF.loc[whDF.wcount==9]
whDF10 = whDF.loc[whDF.wcount==10]
whDF11 = whDF.loc[whDF.wcount==11]
whDF12 = whDF.loc[whDF.wcount==12]
whDF13 = whDF.loc[whDF.wcount==13]

allwhDFs = [whDF0,whDF1,whDF2,whDF3,whDF4,whDF5,whDF6,whDF7,whDF8,whDF9,whDF10,whDF11,whDF12,whDF13]

#xvalues to plot
xplot = np.arange(0,max(whDF.twake))

##Plot
sfig5A = plt.figure(figsize=(12,4))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for idx,subDF in enumerate(allwhDFs[1:11]):
    plt.subplot(2,7,idx+1)
    if len(subDF)>1:
        x1 = subDF.twake
        x1 = sm.add_constant(x1)
        y1 = subDF.sws
        mod1 = sm.OLS(y1,x1)
        res1 = mod1.fit()
        linreg1 = lambda x: res1.params[1]*x + res1.params[0]
        plt.plot(xplot, linreg1(xplot), color='red', ls='--')
        #plt.text(1000,2800,s='r2='+str(np.round(res1.rsquared,4)))
        #plt.text(1000,2500,s='p='+str(np.round(res1.pvalues[1],7)))
        print(res1.rsquared)
    plt.scatter(subDF.twake, subDF.sws, s=10, color='gray', alpha=0.4)
    sns.despine()
    plt.ylim([0,3010])
    plt.xlim([0,max(whDF.twake)+5])
    plt.xticks([0,5000],["",""])
    plt.yticks([0,1000,2000,3000],["","","",""])

sfig5A.savefig(outpath+'sf5A_nrembywake.png')



#Split data by total wake
wwDF100 = whDF.loc[whDF.twake<100]
wwDF200 = whDF.loc[(whDF.twake>=100)&(whDF.twake<200)]
wwDF300 = whDF.loc[(whDF.twake>=200)&(whDF.twake<300)]
wwDF400 = whDF.loc[(whDF.twake>=300)&(whDF.twake<400)]
wwDF500 = whDF.loc[(whDF.twake>=400)&(whDF.twake<500)]
wwDF1000 = whDF.loc[(whDF.twake>=500)&(whDF.twake<1000)]
wwDF1500 = whDF.loc[(whDF.twake>=1000)&(whDF.twake<1500)]
wwDF2000 = whDF.loc[(whDF.twake>=1500)&(whDF.twake<2000)]
wwDF2500 = whDF.loc[(whDF.twake>=2000)&(whDF.twake<2500)]
wwDF3000 = whDF.loc[(whDF.twake>=2500)&(whDF.twake<3000)]
wwDF3500 = whDF.loc[(whDF.twake>=3000)&(whDF.twake<3500)]
wwDF4000 = whDF.loc[(whDF.twake>=3500)&(whDF.twake<4000)]
wwDF4500 = whDF.loc[(whDF.twake>=4000)&(whDF.twake<4500)]
wwDF5000 = whDF.loc[(whDF.twake>=4500)&(whDF.twake<5000)]
wwDF5500 = whDF.loc[(whDF.twake>=5000)&(whDF.twake<5500)]
wwDF6000 = whDF.loc[(whDF.twake>=5500)&(whDF.twake<6000)]
wwDF6500 = whDF.loc[(whDF.twake>=6000)&(whDF.twake<6500)]
wwDF7000 = whDF.loc[(whDF.twake>=6500)&(whDF.twake<7000)]


allwwDFs = [wwDF100,wwDF200,wwDF300,wwDF400,wwDF500,wwDF1000,wwDF1500,wwDF2000,wwDF2500,wwDF3000,wwDF3500,wwDF4000,wwDF4500,wwDF5000,wwDF5500,wwDF6000,wwDF6500,wwDF7000]

xplot2 = np.arange(0,15,0.1)

sfig5B = plt.figure(figsize=(12,4))
plt.subplots_adjust(wspace=0.4,hspace=0.4)
for idx,subDF in enumerate(allwwDFs[0:13]):
    plt.subplot(2,7,idx+1)
    if len(subDF)>1: 
        x1 = subDF.wcount
        x1 = sm.add_constant(x1)
        y1 = subDF.sws
        mod1 = sm.OLS(y1,x1)
        res1 = mod1.fit()
        linreg1 = lambda x: res1.params[1]*x + res1.params[0]
        plt.plot(xplot2, linreg1(xplot2), color='red', ls='--')
        #plt.text(1,2900,s='r2='+str(np.round(res1.rsquared,4)))
        #print(res1.pvalues[1])
        print(res1.rsquared)
    plt.scatter(subDF.wcount, subDF.sws, s=10, color='gray', alpha=0.4)
    sns.despine()
    plt.ylim([0,3010])
    plt.xlim([0,14])
    plt.xticks([0,10],["",""])
    plt.yticks([0,1000,2000,3000],["","","",""])

sfig5B.savefig(outpath+'sf5b_nrembywcount.png')

