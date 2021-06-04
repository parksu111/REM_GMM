#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:14:45 2021

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
import scipy.io as so
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pingouin

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/final_figures/Sfig6/'

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
postDF = postDF.loc[postDF.rem<240]
pnonDF = postDF.loc[postDF.rem<7.5]
psubDF = postDF.loc[postDF.rem>=7.5]
psubDF = psubDF.reset_index(drop=True)


#CDF
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


###############################################################################
#### A - REMpost by inter-REM


#xvalues to plot
xes1 = np.arange(750, 8000, 1500)
xplot1 = np.arange(0,8500)

#Labels for plot
splitlbls1 = ['[0,1500)','[1500,3000)','[3000,4500)','[4500,6000)','[6000,7500)']

inter1500 = psinDF.loc[psinDF.inter<1500]
inter3000 = psinDF.loc[(psinDF.inter>=1500)&(psinDF.inter<3000)]
inter4500 = psinDF.loc[(psinDF.inter>=3000)&(psinDF.inter<4500)]
inter6000 = psinDF.loc[(psinDF.inter>=4500)&(psinDF.inter<6000)]
inter7500 = psinDF.loc[(psinDF.inter>=6000)&(psinDF.inter<7500)]

intrpost = [inter1500.rpost,inter3000.rpost,inter4500.rpost,inter6000.rpost,inter7500.rpost]

x1 = psinDF.inter
x1 = sm.add_constant(x1)
y1 = psinDF.rpost
mod1 = sm.OLS(y1,x1)
res1 = mod1.fit()
linreg1 = lambda x: res1.params[1]*x+res1.params[0]

sfig6a = plt.figure(figsize=(7.5,6))
bp = plt.boxplot(intrpost, positions=xes1, widths=1500/3, vert=True, patch_artist=True,zorder=1)
for patch in bp['boxes']:
    patch.set_facecolor('gray')
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot1, linreg1(xplot1), color='red', ls='--', zorder=2)
plt.xticks(xes1,splitlbls1)
sns.despine()
plt.ylim([0,310])


sfig6a.savefig(outpath+'interrpost.pdf')

###############################################################################
#### B - REM spectral density (parietal)

#Lists to store results
nperc20_1 = []
nperc40_1 = []
nperc60_1 = []
nperc80_1 = []
nperc100_1 = []

twin=3

#Loop through recordings and calculate power spectra of REMpost based on CDF
for rec in recordings:
    #Load recordings and find cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    sr = rp.get_snr(ppath, rec)
    nbin = int(np.round(sr)*2.5)
    dt = nbin*1/sr
    nwin = np.round(twin*sr)
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG.mat'))['EEG']).astype('float')
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
                        nperc20_1.extend(pows)
                    elif cdf<0.4:
                        nperc40_1.extend(pows)
                    elif cdf<0.6:
                        nperc60_1.extend(pows)
                    elif cdf<0.8:
                        nperc80_1.extend(pows)
                    else:
                        nperc100_1.extend(pows)
            cnt2+=1
    print(rec)

#Compile results
allnPercs_1 = [nperc20_1,nperc40_1,nperc60_1,nperc80_1,nperc100_1]

#Limit frequency to [0,15]Hz
ifreq2 = np.where(F<=15)
F2 = F[ifreq2]
subS20_1 = np.array([x[ifreq2] for x in nperc20_1])
subS40_1 = np.array([x[ifreq2] for x in nperc40_1])
subS60_1 = np.array([x[ifreq2] for x in nperc60_1])
subS80_1 = np.array([x[ifreq2] for x in nperc80_1])
subS100_1 = np.array([x[ifreq2] for x in nperc100_1])

#Plot
fig6D = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=7)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,spec in enumerate(allnPercs_1):
    subS = np.array([x[ifreq2] for x in spec])
    colorVal = scalarMap.to_rgba(idx+2)
    lbl = str(idx*20*0.01)+'<=cdfval<'+str((idx+1)*20*0.01) 
    plt.plot(F2, subS.mean(axis=0), color=colorVal, label=lbl, lw=1)
sns.despine()
plt.legend()

fig6D.savefig(outpath+'sf6_parietal_cdf.pdf')


#Stats
deltafreq = np.where((F>=0.5)&(F<=4.5))
thetafreq = np.where((F>=5)&(F<=9.5))
sigmafreq = np.where((F>=10)&(F<=15))
df = F[1]-F[0]

deltapows_1 = []
thetapows_1 = []
sigmapows_1 = []
cdfgroup_1 = []

for idx,subpows in enumerate(allnPercs_1):
    for subpow in subpows:
        deltapows_1.append(np.sum(subpow[deltafreq])*df)
        thetapows_1.append(np.sum(subpow[thetafreq])*df)
        sigmapows_1.append(np.sum(subpow[sigmafreq])*df)
        if idx==0:
            cdfgroup_1.append(20)
        elif idx==1:
            cdfgroup_1.append(40)
        elif idx==2:
            cdfgroup_1.append(60)
        elif idx==3:
            cdfgroup_1.append(80)
        else:
            cdfgroup_1.append(100)

cdfpowDF_1 = pd.DataFrame(list(zip(deltapows_1,thetapows_1,sigmapows_1,cdfgroup_1)),columns=['delta','theta','sigma','cdfg'])

#delta
pingouin.homoscedasticity(data=cdfpowDF_1, dv="delta", group="cdfg")
pingouin.anova(data=cdfpowDF_1, dv="delta", between="cdfg")
#theta
pingouin.homoscedasticity(data=cdfpowDF_1, dv="theta", group="cdfg")
pingouin.welch_anova(data=cdfpowDF_1, dv="theta", between="cdfg")
#sigma
pingouin.homoscedasticity(data=cdfpowDF_1, dv="sigma", group="cdfg")
pingouin.welch_anova(data=cdfpowDF_1, dv="sigma", between="cdfg")
###############################################################################

#### C - Compare Actual spectra to weighted average spectra

#Remove recording with no EEG2
recordings2 = [x for x in recordings if x!='HA41_081618n1']


##Store spectra of REM based on REMpre
remspec30 = []
remspec60 = []
remspec90 = []
remspec120 = []
remspec150 = []
remspec180 = []
remspec210 = []
remspec240 = []

twin=3

for rec in recordings2:
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
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            states = np.array([x[0] for x in sub])
            remInds = np.where((states=='R'))[0]
            remSeqs = rp.stateSeq(sub, remInds)
            
            remSpec = []
            
            for s in remSeqs:
                if len(s)*nbin>=nwin:
                    b = int((s[-1]+1)*nbin)
                    sup = list(range(int(s[0]*nbin), b))
                    if sup[-1]>len(EEG):
                        sup = list(range(int(s[0]*nbin), len(EEG)))
                    if len(sup) >= nwin:
                        Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                        remSpec.append(Pow)      
            rem = sub[0][1]*2.5
            if (rem>=7.5)&(rem<240):
                if rem<30:
                    remspec30.extend(remSpec)
                elif rem<60:
                    remspec60.extend(remSpec)
                elif rem<90:
                    remspec90.extend(remSpec)                    
                elif rem<120:
                    remspec120.extend(remSpec)
                elif rem<150:
                    remspec150.extend(remSpec)                    
                elif rem<180:
                    remspec180.extend(remSpec)
                elif rem<210:
                    remspec210.extend(remSpec)                    
                else:
                    remspec240.extend(remSpec)
            cnt2+=1
    print(rec)

# Calculate proportions
psub20 = psinDF.loc[psinDF.cdf<0.2]
psub40 = psinDF.loc[(psinDF.cdf>=0.2)&(psinDF.cdf<0.4)]
psub60 = psinDF.loc[(psinDF.cdf>=0.4)&(psinDF.cdf<0.6)]
psub80 = psinDF.loc[(psinDF.cdf>=0.6)&(psinDF.cdf<0.8)]
psub100 = psinDF.loc[psinDF.cdf>=0.8]

psub20Splits = rp.splitData(psub20,30)
psub40Splits = rp.splitData(psub40,30)
psub60Splits = rp.splitData(psub60,30)
psub80Splits = rp.splitData(psub80,30)
psub100Splits = rp.splitData(psub100,30)

c20r30 = len(psub20Splits[0])/len(psub20)
c20r60 = len(psub20Splits[1])/len(psub20)
c20r90 = len(psub20Splits[2])/len(psub20)
c20r120 = len(psub20Splits[3])/len(psub20)
c20r150 = len(psub20Splits[4])/len(psub20)
c20r180 = len(psub20Splits[5])/len(psub20)
c20r210 = len(psub20Splits[6])/len(psub20)
c20r240 = len(psub20Splits[7])/len(psub20)

c40r30 = len(psub40Splits[0])/len(psub40)
c40r60 = len(psub40Splits[1])/len(psub40)
c40r90 = len(psub40Splits[2])/len(psub40)
c40r120 = len(psub40Splits[3])/len(psub40)
c40r150 = len(psub40Splits[4])/len(psub40)
c40r180 = len(psub40Splits[5])/len(psub40)
c40r210 = len(psub40Splits[6])/len(psub40)
c40r240 = len(psub40Splits[7])/len(psub40)

c60r30 = len(psub60Splits[0])/len(psub60)
c60r60 = len(psub60Splits[1])/len(psub60)
c60r90 = len(psub60Splits[2])/len(psub60)
c60r120 = len(psub60Splits[3])/len(psub60)
c60r150 = len(psub60Splits[4])/len(psub60)
c60r180 = len(psub60Splits[5])/len(psub60)
c60r210 = len(psub60Splits[6])/len(psub60)
c60r240 = len(psub60Splits[7])/len(psub60)

c80r30 = len(psub80Splits[0])/len(psub80)
c80r60 = len(psub80Splits[1])/len(psub80)
c80r90 = len(psub80Splits[2])/len(psub80)
c80r120 = len(psub80Splits[3])/len(psub80)
c80r150 = len(psub80Splits[4])/len(psub80)
c80r180 = len(psub80Splits[5])/len(psub80)
c80r210 = len(psub80Splits[6])/len(psub80)
c80r240 = len(psub80Splits[7])/len(psub80)

c100r30 = len(psub100Splits[0])/len(psub100)
c100r60 = len(psub100Splits[1])/len(psub100)
c100r90 = len(psub100Splits[2])/len(psub100)
c100r120 = len(psub100Splits[3])/len(psub100)
c100r150 = len(psub100Splits[4])/len(psub100)
c100r180 = len(psub100Splits[5])/len(psub100)
c100r210 = len(psub100Splits[6])/len(psub100)
c100r240 = len(psub100Splits[7])/len(psub100)


#Limit frequency to [0,15]Hz
df = F[1]-F[0]
ifreq2 = np.where(F<=15)
F2 = F[ifreq2]

sub30 = np.array([x[ifreq2] for x in remspec30])
sub60 = np.array([x[ifreq2] for x in remspec60])
sub90 = np.array([x[ifreq2] for x in remspec90])
sub120 = np.array([x[ifreq2] for x in remspec120])
sub150 = np.array([x[ifreq2] for x in remspec150])
sub180 = np.array([x[ifreq2] for x in remspec180])
sub210 = np.array([x[ifreq2] for x in remspec210])
sub240 = np.array([x[ifreq2] for x in remspec240])


## Calculate spectra based on CDF
nperc20 = []
nperc40 = []
nperc60 = []
nperc80 = []
nperc100 = []

for rec in recordings2:
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
            if (rem>=7.5)&(rem<240):
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

#Limit Frequency to [0,15]Hz
subS20 = np.array([x[ifreq2] for x in nperc20])
subS40 = np.array([x[ifreq2] for x in nperc40])
subS60 = np.array([x[ifreq2] for x in nperc60])
subS80 = np.array([x[ifreq2] for x in nperc80])
subS100 = np.array([x[ifreq2] for x in nperc100])

#Calculate 99% C.I.
aspec20 = subS20.mean(axis=0) - 2.576*subS20.std(axis=0)/np.sqrt(len(subS20))
bspec20 = subS20.mean(axis=0) + 2.576*subS20.std(axis=0)/np.sqrt(len(subS20))
aspec40 = subS40.mean(axis=0) - 2.576*subS40.std(axis=0)/np.sqrt(len(subS40))
bspec40 = subS40.mean(axis=0) + 2.576*subS40.std(axis=0)/np.sqrt(len(subS40))
aspec60 = subS60.mean(axis=0) - 2.576*subS60.std(axis=0)/np.sqrt(len(subS60))
bspec60 = subS60.mean(axis=0) + 2.576*subS60.std(axis=0)/np.sqrt(len(subS60))
aspec80 = subS80.mean(axis=0) - 2.576*subS80.std(axis=0)/np.sqrt(len(subS80))
bspec80 = subS80.mean(axis=0) + 2.576*subS80.std(axis=0)/np.sqrt(len(subS80))
aspec100 = subS100.mean(axis=0) - 2.576*subS100.std(axis=0)/np.sqrt(len(subS100))
bspec100 = subS100.mean(axis=0) + 2.576*subS100.std(axis=0)/np.sqrt(len(subS100))

#Plot
sfig6D = plt.figure(figsize=(15,6))
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=7)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.subplot(1,5,1)
plt.plot(F2, c20r30*sub30.mean(axis=0)+c20r60*sub60.mean(axis=0)+c20r90*sub90.mean(axis=0)+c20r120*sub120.mean(axis=0)+c20r150*sub150.mean(axis=0)+c20r180*sub180.mean(axis=0)+c20r210*sub210.mean(axis=0)+c20r240*sub240.mean(axis=0), color=scalarMap.to_rgba(2),ls='--',lw=1)
subS = np.array([x[ifreq2] for x in nperc20])
plt.plot(F2, subS.mean(axis=0), color=scalarMap.to_rgba(2),lw=1)
plt.fill_between(F2, aspec20, bspec20, color=scalarMap.to_rgba(2), alpha=0.2)
sns.despine()
plt.ylim([0,600])
plt.subplot(1,5,2)
plt.plot(F2, c40r30*sub30.mean(axis=0)+c40r60*sub60.mean(axis=0)+c40r90*sub90.mean(axis=0)+c40r120*sub120.mean(axis=0)+c40r150*sub150.mean(axis=0)+c40r180*sub180.mean(axis=0)+c40r210*sub210.mean(axis=0)+c40r240*sub240.mean(axis=0), color=scalarMap.to_rgba(3),ls='--',lw=1)
subS = np.array([x[ifreq2] for x in nperc40])
plt.plot(F2, subS.mean(axis=0), color=scalarMap.to_rgba(3),lw=1)
plt.fill_between(F2, aspec40, bspec40, color=scalarMap.to_rgba(3), alpha=0.2)
sns.despine()
plt.ylim([0,600])
plt.subplot(1,5,3)
plt.plot(F2, c60r30*sub30.mean(axis=0)+c60r60*sub60.mean(axis=0)+c60r90*sub90.mean(axis=0)+c60r120*sub120.mean(axis=0)+c60r150*sub150.mean(axis=0)+c60r180*sub180.mean(axis=0)+c60r210*sub210.mean(axis=0)+c60r240*sub240.mean(axis=0), color=scalarMap.to_rgba(4),ls='--',lw=1)
subS = np.array([x[ifreq2] for x in nperc60])
plt.plot(F2, subS.mean(axis=0), color=scalarMap.to_rgba(4),lw=1)
plt.fill_between(F2, aspec60, bspec60, color=scalarMap.to_rgba(4), alpha=0.2)
sns.despine()
plt.ylim([0,600])
plt.subplot(1,5,4)
plt.plot(F2, c80r30*sub30.mean(axis=0)+c80r60*sub60.mean(axis=0)+c80r90*sub90.mean(axis=0)+c80r120*sub120.mean(axis=0)+c80r150*sub150.mean(axis=0)+c80r180*sub180.mean(axis=0)+c80r210*sub210.mean(axis=0)+c80r240*sub240.mean(axis=0), color=scalarMap.to_rgba(5),ls='--',lw=1)
subS = np.array([x[ifreq2] for x in nperc80])
plt.plot(F2, subS.mean(axis=0), color=scalarMap.to_rgba(5),lw=1)
plt.fill_between(F2, aspec80, bspec80, color=scalarMap.to_rgba(5), alpha=0.2)
sns.despine()
plt.ylim([0,600])
plt.subplot(1,5,5)
plt.plot(F2, c100r30*sub30.mean(axis=0)+c100r60*sub60.mean(axis=0)+c100r90*sub90.mean(axis=0)+c100r120*sub120.mean(axis=0)+c100r150*sub150.mean(axis=0)+c100r180*sub180.mean(axis=0)+c100r210*sub210.mean(axis=0)+c100r240*sub240.mean(axis=0), color=scalarMap.to_rgba(6),ls='--',lw=1)
subS = np.array([x[ifreq2] for x in nperc100])
plt.plot(F2, subS.mean(axis=0), color=scalarMap.to_rgba(6),lw=1)
plt.fill_between(F2, aspec100, bspec100, color=scalarMap.to_rgba(6), alpha=0.2)
sns.despine()
plt.ylim([0,600])

sfig6D.savefig(outpath+'sf6d_actualvswavg.pdf')


###############################################################################
#### D - Compare REMpost of single cycles with wake without wake


#### Bootstrap
psinDF.reset_index(drop=True)

all_ww_params = []
all_wo_params = []

ww_params60 = []
wo_params60 = []
ww_params120 = []
wo_params120 = []
ww_params180 = []
wo_params180 = []
ww_params240 = []
wo_params240 = []

for i in range(10000):
    np.random.seed(i)
    sample = np.random.choice(3818,3818)
    cdfs = []
    rposts = []
    rpres = []
    wakes = []
    for x in sample:
        row = psinDF.iloc[x]
        rem = row['rem']
        rpost = row['rpost']
        cdf = row['cdf']
        wake = row['wake']
        cdfs.append(cdf)
        rposts.append(rpost)
        rpres.append(rem)
        wakes.append(wake)
    bootDF = pd.DataFrame(list(zip(rpres,rposts,cdfs,wakes)),columns=['rem','rpost','cdf','wake'])
    wwDF = bootDF.loc[bootDF.wake>0]
    woDF = bootDF.loc[bootDF.wake==0]
    wwDF.reset_index(drop=True)
    woDF.reset_index(drop=True)
    #ww
    x1 = wwDF.cdf
    x1 = sm.add_constant(x1)
    y1 = wwDF.rpost
    mod1 = sm.OLS(y1,x1)
    res1 = mod1.fit()
    all_ww_params.append((res1.params[1],res1.params[0]))
    #wo
    x2 = woDF.cdf
    x2 = sm.add_constant(x2)
    y2 = woDF.rpost
    mod2 = sm.OLS(y2,x2)
    res2 = mod2.fit()
    all_wo_params.append((res2.params[1],res2.params[0]))
    splitDFs = rp.splitData(bootDF, 60)
    for idx,subDF in enumerate(splitDFs):
        subwwdf = subDF.loc[subDF.wake>0]
        subwodf = subDF.loc[subDF.wake==0]
        subwwdf.reset_index(drop=True)
        subwodf.reset_index(drop=True)
        #ww
        x3 = subwwdf.cdf
        x3 = sm.add_constant(x3)
        y3 = subwwdf.rpost
        mod3 = sm.OLS(y3,x3)
        res3 = mod3.fit()
        #wo
        x4 = subwodf.cdf
        x4 = sm.add_constant(x4)
        y4 = subwodf.rpost
        mod4 = sm.OLS(y4,x4)
        res4 = mod4.fit()
        if idx==0:
            ww_params60.append((res3.params[1],res3.params[0]))
            wo_params60.append((res4.params[1],res4.params[0]))
        elif idx==1:
            ww_params120.append((res3.params[1],res3.params[0]))
            wo_params120.append((res4.params[1],res4.params[0]))
        elif idx==2:
            ww_params180.append((res3.params[1],res3.params[0]))
            wo_params180.append((res4.params[1],res4.params[0]))
        else:
            ww_params240.append((res3.params[1],res3.params[0]))
            wo_params240.append((res4.params[1],res4.params[0]))
    print(i)
    

xes = np.arange(0,1,0.01)

    
allwithwake = [all_ww_params, ww_params60, ww_params120, ww_params180, ww_params240]
allwowake = [all_wo_params, wo_params60, wo_params120, wo_params180, wo_params240]



#xvalues to plot
xes4 = np.arange(0.1,1,0.2)
xplot4 = np.arange(0,1,0.01)

lbls = ['[7.5,60)','[60,120)','[120,180)','[180,240)']

psinwwDF = psinDF.loc[psinDF.wake>0]
psinwoDF = psinDF.loc[psinDF.wake==0]

psinwwDF.reset_index(drop=True)
psinwoDF.reset_index(drop=True)

wwcdf20 = psinwwDF.loc[psinwwDF.cdf<0.2]
wwcdf40 = psinwwDF.loc[(psinwwDF.cdf>=0.2)&(psinwwDF.cdf<0.4)]
wwcdf60 = psinwwDF.loc[(psinwwDF.cdf>=0.4)&(psinwwDF.cdf<0.6)]
wwcdf80 = psinwwDF.loc[(psinwwDF.cdf>=0.6)&(psinwwDF.cdf<0.8)]
wwcdf100 = psinwwDF.loc[psinwwDF.cdf>=0.8]

wocdf20 = psinwoDF.loc[psinwoDF.cdf<0.2]
wocdf40 = psinwoDF.loc[(psinwoDF.cdf>=0.2)&(psinwoDF.cdf<0.4)]
wocdf60 = psinwoDF.loc[(psinwoDF.cdf>=0.4)&(psinwoDF.cdf<0.6)]
wocdf80 = psinwoDF.loc[(psinwoDF.cdf>=0.6)&(psinwoDF.cdf<0.8)]
wocdf100 = psinwoDF.loc[psinwoDF.cdf>=0.8]

x1 = psinwwDF.cdf
x1 = sm.add_constant(x1)
y1 = psinwwDF.rpost
mod1 = sm.OLS(y1,x1)
res1 = mod1.fit()
linreg1 = lambda x: res1.params[1]*x + res1.params[0]

x2 = psinwoDF.cdf
x2 = sm.add_constant(x2)
y2 = psinwoDF.rpost
mod2 = sm.OLS(y2,x2)
res2 = mod2.fit()
linreg2 = lambda x: res2.params[1]*x + res2.params[0]

wwsplitDFs = rp.splitData(psinwwDF,60)
wosplitDFs = rp.splitData(psinwoDF,60)



sfig5B = plt.figure(figsize=(20,18))
cNorm = colors.Normalize(vmin=0, vmax=7)
cmap = plt.get_cmap('YlOrRd')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.subplot(3,5,1)
slope = np.round(res1.params[1],6)
pval = np.round(res1.pvalues[1],6)
plt.text(0.3,295,s='slope='+str(slope))
plt.text(0.3,280,s='pval='+str(pval))
bp = plt.boxplot([wwcdf20.rpost,wwcdf40.rpost,wwcdf60.rpost,wwcdf80.rpost,wwcdf100.rpost],positions=xes4, vert=True, patch_artist=True, zorder=1)
for ind,patch in enumerate(bp['boxes']):
    colorVal = scalarMap.to_rgba(ind+2)
    patch.set_facecolor(colorVal)
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot4,linreg1(xplot4), color='k',ls='--',zorder=2)
sns.despine()
plt.xticks(xes4,[1,2,3,4,5])
plt.xlim([0,1])
plt.ylim([0,310])
for idx,subDF in enumerate(wwsplitDFs):
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
    plt.subplot(3,5,idx+2)
    slope = np.round(res.params[1],6)
    pval = np.round(res.pvalues[1],6)
    plt.text(0.3,295,s='slope='+str(slope))
    plt.text(0.3,280,s='pval='+str(pval))
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
    print(res.pvalues)
plt.subplot(3,5,6)
slope = np.round(res2.params[1],6)
pval = np.round(res2.pvalues[1],6)
plt.text(0.3,295,s='slope='+str(slope))
plt.text(0.3,280,s='pval='+str(pval))
bp = plt.boxplot([wocdf20.rpost,wocdf40.rpost,wocdf60.rpost,wocdf80.rpost,wocdf100.rpost],positions=xes4, vert=True, patch_artist=True, zorder=1)
for ind,patch in enumerate(bp['boxes']):
    colorVal = scalarMap.to_rgba(ind+2)
    patch.set_facecolor(colorVal)
for patch in bp['medians']:
    plt.setp(patch, color='black')
plt.plot(xplot4,linreg2(xplot4), color='k',ls='--',zorder=2)
sns.despine()
plt.xticks(xes4,[1,2,3,4,5])
plt.xlim([0,1])
plt.ylim([0,310])
for idx,subDF in enumerate(wosplitDFs):
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
    plt.subplot(3,5,idx+7)
    slope = np.round(res.params[1],6)
    pval = np.round(res.pvalues[1],6)
    plt.text(0.3,295,s='slope='+str(slope))
    plt.text(0.3,280,s='pval='+str(pval))    
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
    print(res.pvalues)
for idx,subparams in enumerate(allwithwake):
    plt.subplot(3,5,idx+11)
    subparams2 = allwowake[idx]
    wwres = []
    for params in subparams:
        linreg = lambda x: params[1]*x + params[0]
        wwres.append(linreg(xes))
    wores = []
    for params in subparams2:
        linreg = lambda x: params[1]*x + params[0]
        wores.append(linreg(xes))
    ww2_5 = np.percentile(wwres,2.5,axis=0)
    ww97_5 = np.percentile(wwres,97.5,axis=0)
    wwmean = np.mean(wwres,axis=0)
    wo2_5 = np.percentile(wores,2.5,axis=0)
    wo97_5 = np.percentile(wores,97.5,axis=0)
    womean = np.mean(wores,axis=0)
    plt.plot(xes,wwmean,color='purple',label='with wake')
    plt.fill_between(xes,ww2_5,ww97_5,color='purple',alpha=0.3)
    plt.plot(xes,womean,color='orange', label='w/o wake')
    plt.fill_between(xes,wo2_5,wo97_5,color='orange',alpha=0.3)
    sns.despine()
    plt.legend()
    plt.xlim([0,1])
    plt.ylim([0,310])
    
sfig5B.savefig(outpath+'sf5b_rpostcdf_ww_wo.pdf')

    



