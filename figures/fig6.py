#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:06:09 2021

@author: cwlab08
"""

#### Preparations
import os
import rempropensity as rp
import statsmodels.api as sm
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.optimize import fsolve
from scipy import stats
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.io as so
import pingouin


#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/REM_GMM/final_figures/6_RemDur/'

#Define colors
current_palette = sns.color_palette('muted', 10)
col1 = current_palette[3]
col2 = current_palette[0]

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


# Make REM-NREM cycle dataframe including REMpost
postDF = rp.post_recToDF(ppath, recordings, 8, False)
pnonDF = postDF.loc[postDF.rem<7.5]
psubDF = postDF.loc[(postDF.rem>=7.5)&(postDF.rem<240)]
psubDF = psubDF.reset_index(drop=True)

#Calculate CDF at point of transition from NREM to REM
cdfvals = []
for idx,row in psubDF.iterrows():
    rem = row['rem']
    sws = row['sws']
    logsws = np.log(sws)
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    s2 = slow_lina*rem+slow_linb
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    if k2==0:
        f = lambda x: k1*stats.norm.cdf(x,m1,s1)
    else:
        f = lambda x: k1*stats.norm.cdf(x,m1,s1)+k2*stats.norm.cdf(x,m2,s2)
    cdf = f(logsws)
    cdfvals.append(cdf)
    print(idx)

#Add cdf to dataframe
psubDF['cdf'] = cdfvals


# Divide REMpost Dataframe into single and sequential
pseqrows = []
psinrows = []

for index,row in psubDF.iterrows():
    rem = row['rem']
    sws = row['sws']
    if rem > max(intersect_x):
        psinrows.append(row)
    else:
        indBurst = np.where(intersect_x==rem)[0][0]
        burstLim = exp_intersect_y[indBurst][0]
        if sws<burstLim:
            pseqrows.append(row)
        else:
            psinrows.append(row)

pseqDF = pd.DataFrame(pseqrows)
psinDF = pd.DataFrame(psinrows)

pseqDF = pseqDF.reset_index(drop=True)
psinDF = psinDF.reset_index(drop=True)

###############################################################################
#### Fig 6A - REMpost by REMpre

#xvalues to plot
xes = np.arange(15,240,30)
xplot = np.arange(0,240)

#Labels for plot
splitlbls = ['[0,30)','[30,60)','[60,90)','[90,120)','[120,150)','[150,180)','[180,210)','[210,240)']

### All cycles

x = psubDF.rem
x = sm.add_constant(x)
y = psubDF.rpost
mod = sm.OLS(y,x)
res = mod.fit()


###(Left) Sequential

#Split data by 30s bins of REMpre (Sequential cycles)
splitSeqDFs = rp.splitData(pseqDF)
split_seqrposts = [x.rpost for x in splitSeqDFs]

#Linear Regression: REMpre vs. REMpost
x1 = pseqDF.rem
x1 = sm.add_constant(x1)
y1 = pseqDF.rpost
mod1 = sm.OLS(y1,x1)
res1 = mod1.fit()
linreg1 = lambda x: res1.params[1]*x + res1.params[0]

#Plot
fig6A1 = plt.figure(figsize=(6.25,6))
bp = plt.boxplot(split_seqrposts, positions=xes, widths=10, vert=True, patch_artist=True,zorder=1)
for patch in bp['boxes']:
    patch.set_facecolor(col2)
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot, linreg1(xplot), color='red', ls='--', zorder=2)
plt.xticks(xes,splitlbls)
sns.despine()
plt.xlim([0,150])
plt.ylim([0,310])

fig6A1.savefig(outpath+'f6a_seqremrpost.pdf')


###(Right) Single

#Split data by 30s bins of REMpre (Single cycles)
splitSinDFs = rp.splitData(psinDF)
split_sinrposts = [x.rpost for x in splitSinDFs]

#Linear Regression: REMpre vs. REMpost
x2 = psinDF.rem
x2 = sm.add_constant(x2)
y2 = psinDF.rpost
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()
linreg2 = lambda x: res2.params[1]*x + res2.params[0]

fig6A2 = plt.figure(figsize=(10,6))
bp = plt.boxplot(split_sinrposts, positions=xes, widths=10, vert=True, patch_artist=True,zorder=1)
for patch in bp['boxes']:
    patch.set_facecolor(col1)
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot, linreg2(xplot), color='red', ls='--', zorder=2)
plt.xticks(xes,splitlbls)
sns.despine()
plt.xlim([0,240])
plt.ylim([0,310])

fig6A2.savefig(outpath+'f6a_sinremrpost.pdf')



###############################################################################
#### Fig 6B - REMpost by |N|

#xvalues to plot
xes2 = np.arange(250,2750,500)
xplot2 = np.arange(0,3010)

#Linear Regression
x2 = psinDF.sws
x2 = sm.add_constant(x2)
y2 = psinDF.rpost
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()
linreg2 = lambda x: res2.params[1]*x + res2.params[0]

#Split data by |N|
nrem500 = psinDF.loc[psinDF.sws<500]
nrem1000 = psinDF.loc[(psinDF.sws>=500)&(psinDF.sws<1000)]
nrem1500 = psinDF.loc[(psinDF.sws>=1000)&(psinDF.sws<1500)]
nrem2000 = psinDF.loc[(psinDF.sws>=1500)&(psinDF.sws<2000)]
nrem2500 = psinDF.loc[(psinDF.sws>=2000)&(psinDF.sws<2500)]

splitDFs2 = [nrem500,nrem1000,nrem1500,nrem2000,nrem2500]
splitRposts2 = [x.rpost for x in splitDFs2]

#Plot
fig6B = plt.figure()
bp = plt.boxplot(splitRposts2, positions=xes2, widths=500/3, vert=True, patch_artist=True, zorder=1)
for patch in bp['boxes']:
    patch.set_facecolor('gray')
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot2, linreg2(xplot2), color='red', ls='--', zorder=2)
sns.despine()
plt.ylim([0,310])

fig6B.savefig(outpath+'f6b_rempostbynrem.pdf')


###############################################################################
#### Fig 6C - REMpost by CDF

###(Left) For all values of REMpre

#Split dataframe by CDF (20%)
cdf20 = psinDF.loc[psinDF.cdf<0.2]
cdf40 = psinDF.loc[(psinDF.cdf>=0.2)&(psinDF.cdf<0.4)]
cdf60 = psinDF.loc[(psinDF.cdf>=0.4)&(psinDF.cdf<0.6)]
cdf80 = psinDF.loc[(psinDF.cdf>=0.6)&(psinDF.cdf<0.8)]
cdf100 = psinDF.loc[psinDF.cdf>=0.8]

#Linear Regression: CDF vs. REMpost
x5 = psinDF.cdf
y5 = psinDF.rpost
x5 = sm.add_constant(x5)
mod5 = sm.OLS(y5,x5)
res5 = mod5.fit()
linreg5 = lambda x: res5.params[1]*x + res5.params[0]

#Split data by 60s of REMpre
psubsplitDFs = rp.splitData(psinDF,60)

#xvalues to plot
xes4 = np.arange(0.1,1,0.2)
xplot4 = np.arange(0,1,0.01)

#plot
fig6C = plt.figure(figsize=(15,6))
cNorm = colors.Normalize(vmin=0, vmax=7)
cmap = plt.get_cmap('YlOrRd')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.subplot(1,5,1)
bp = plt.boxplot([cdf20.rpost,cdf40.rpost,cdf60.rpost,cdf80.rpost,cdf100.rpost],positions=xes4, vert=True, patch_artist=True, zorder=1)
for ind,patch in enumerate(bp['boxes']):
    colorVal = scalarMap.to_rgba(ind+2)
    patch.set_facecolor(colorVal)
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot4,linreg5(xplot4), color='k',ls='--',zorder=2)
sns.despine()
plt.xticks(xes4,['[0,0.2)','[0.2,0.4)','[0.4,0.6)','[0.6,0.8)','[0.8,1.0)'])
plt.xlim([0,1])
plt.ylim([0,310])
for idx,subDF in enumerate(psubsplitDFs):
    s20 = subDF.loc[subDF.cdf<0.2]
    s40 = subDF.loc[(subDF.cdf>=0.2)&(subDF.cdf<0.4)]
    s60 = subDF.loc[(subDF.cdf>=0.4)&(subDF.cdf<0.6)]
    s80 = subDF.loc[(subDF.cdf>=0.6)&(subDF.cdf<0.8)]
    s100 = subDF.loc[subDF.cdf>=0.8]
    x = subDF.cdf
    y = subDF.rpost
    x = sm.add_constant(x)
    mod = sm.OLS(y,x)
    res = mod.fit()
    linreg = lambda x: res.params[1]*x+res.params[0]
    plt.subplot(1,5,idx+2)
    pldatas = [s20.rpost,s40.rpost,s60.rpost,s80.rpost,s100.rpost]
    bp = plt.boxplot(pldatas,positions=xes4, zorder=1, vert=True, patch_artist=True)
    for ind,patch in enumerate(bp['boxes']):
        colorVal = scalarMap.to_rgba(ind+2)
        patch.set_facecolor(colorVal)
    for patch in bp['medians']:
        plt.setp(patch, color='black')
    sns.despine()
    plt.plot(xplot4, linreg(xplot4), color='k', ls='--', zorder=2)
    plt.xlim([0,1])
    plt.ylim([0,310])
    plt.xticks(xes4,[1,2,3,4,5])
    print('s='+str(res.params))
    print('p='+str(res.pvalues))
    print('r2='+str(res.rsquared))

fig6C.savefig(outpath+'f6C_rpostbycdf.pdf')


# Comparison of correlations done via cocor in R
x = psinDF.cdf
y = psinDF.rpost
mod = sm.OLS(y,x)
res = mod.fit()

x2 = psinDF.sws
y2 = psinDF.rpost
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()

x3 = psinDF.cdf
y3 = psinDF.sws
mod3 = sm.OLS(y3,x3)
res3 = mod3.fit()

np.sqrt(res.rsquared)
np.sqrt(res2.rsquared)
np.sqrt(res3.rsquared)

###############################################################################
#### Fig 6D - REM spectra by CDF

#Remove recording that does not have EEG2
recordings2 = [x for x in recordings if x!='HA41_081618n1']

#Lists to store results
nperc20 = []
nperc40 = []
nperc60 = []
nperc80 = []
nperc100 = []

twin=3

#Loop through recordings and calculate power spectra of REMpost based on CDF
for rec in recordings2:
    #Load recordings and find cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    sr = rp.get_snr(ppath, rec)
    nbin = int(np.round(sr)*2.5)
    dt = nbin*1/sr
    nwin = np.round(twin*sr)
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'))['EEG2']).astype('float')
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
        while cnt2 < len(nrt_locs)-2:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]+1]
            rem = sub[0][1]*2.5
            if (rem>=7.5)&(rem<240): #Limit to cycles with REM in [7.5,240)
                #Calculate power spectra
                pows = []
                seqstart = sub[-1][2]-sub[-1][1]+1
                seqend = sub[-1][2]
                s = np.arange(seqstart,seqend+1)
                if len(s)*nbin>=nwin:
                    b = int((s[-1]+1)*nbin)
                    sup = list(range(int(s[0]*nbin),b))
                    if sup[-1]>len(EEG):
                        sup = list(range(int(s[0]*nbin),len(EEG)))
                    if len(sup)>=nwin:
                        Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                        pows.append(Pow)
                #Calculate CDF
                sws = 0
                for x in sub:
                    if (x[0]=='N')or(x[0]=='MA'):
                        sws+=x[1]*2.5
                logsws = np.log(sws)
                k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
                if k1>1:
                    k1=1
                k2 = 1-k1
                m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
                m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
                s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
                s2 = slow_lina*rem+slow_linb
                if k1>0:
                    f = lambda x: k1*stats.norm.cdf(x,m1,s1)+k2*stats.norm.cdf(x,m2,s2)
                else:
                    f = lambda x: k1*stats.norm.cdf(x,m1,s1)
                cdf = f(logsws)
                #Add results
                if rp.isSequential(rem, sws, intersect_x, intersect_y)==False:
                    if cdf<0.2:
                        nperc20.extend(pows)
                    elif cdf<0.4:
                        nperc40.extend(pows)
                    elif cdf<0.6:
                        nperc60.extend(pows)
                    elif cdf<0.8:
                        nperc80.extend(pows)
                    else:
                        nperc100.extend(pows)
            cnt2+=1
    print(rec)

#Compile results
allnPercs = [nperc20,nperc40,nperc60,nperc80,nperc100]

#Limit frequency to [0,15]Hz
ifreq2 = np.where(F<=15)
F2 = F[ifreq2]
subS20 = np.array([x[ifreq2] for x in nperc20])
subS40 = np.array([x[ifreq2] for x in nperc40])
subS60 = np.array([x[ifreq2] for x in nperc60])
subS80 = np.array([x[ifreq2] for x in nperc80])
subS100 = np.array([x[ifreq2] for x in nperc100])

#Plot
fig6D = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=7)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,spec in enumerate(allnPercs):
    subS = np.array([x[ifreq2] for x in spec])
    colorVal = scalarMap.to_rgba(idx+2)
    lbl = str(idx*20*0.01)+'<=cdfval<'+str((idx+1)*20*0.01) 
    plt.plot(F2, subS.mean(axis=0), color=colorVal, label=lbl)
sns.despine()
plt.legend()
plt.ylim([0,650])

fig6D.savefig(outpath+'f6d_remspectra.pdf')


### Statistics

deltafreq = np.where((F>=0.5)&(F<=4.5))
thetafreq = np.where((F>=5)&(F<=9.5))
sigmafreq = np.where((F>=10)&(F<=15))
df = F[1]-F[0]

deltapows = []
thetapows = []
sigmapows = []
cdfgroup = []

for idx,subpows in enumerate(allnPercs):
    for subpow in subpows:
        deltapows.append(np.sum(subpow[deltafreq])*df)
        thetapows.append(np.sum(subpow[thetafreq])*df)
        sigmapows.append(np.sum(subpow[sigmafreq])*df)
        if idx==0:
            cdfgroup.append(20)
        elif idx==1:
            cdfgroup.append(40)
        elif idx==2:
            cdfgroup.append(60)
        elif idx==3:
            cdfgroup.append(80)
        else:
            cdfgroup.append(100)

cdfpowDF = pd.DataFrame(list(zip(deltapows,thetapows,sigmapows,cdfgroup)),columns=['delta','theta','sigma','cdfg'])

#delta
pingouin.homoscedasticity(data=cdfpowDF, dv="delta", group="cdfg")
pingouin.welch_anova(data=cdfpowDF, dv="delta", between="cdfg")
#theta
pingouin.homoscedasticity(data=cdfpowDF, dv="theta", group="cdfg")
pingouin.welch_anova(data=cdfpowDF, dv="theta", between="cdfg")
#sigma
pingouin.homoscedasticity(data=cdfpowDF, dv="sigma", group="cdfg")
pingouin.welch_anova(data=cdfpowDF, dv="sigma", between="cdfg")



