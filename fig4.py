#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:36:43 2020

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
from scipy.optimize import curve_fit
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
outpath = r'/home/cwlab08/Desktop/final_figures/4_RefPerm/'

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


###############################################################################
###Fig 4A (LEFT) - Log Scatterplot with refractory permssive threshold

#x-values to plot
xplot = np.arange(7.5,240,0.1)

fig4A = plt.figure()
plt.scatter(nonDF.rem, nonDF.logsws, s=10, alpha=0.6, color='gray')
plt.scatter(seqDF.rem, seqDF.logsws, s=10, alpha=0.6, color='gray')
plt.scatter(sinDF.rem, sinDF.logsws, s=10, alpha=0.6, color='gray')
plt.plot(np.arange(7.5,242.5,2.5),logthresh1, color='k', lw=1)
plt.axvline(x=60, color='k',ls='--',lw=1)
plt.xticks([0,60,120,180,240],["","","","",""])
plt.yticks([2,4,6,8],["","","",""])
sns.despine()
plt.ylim([2,9])

fig4A.savefig(outpath+'f4A_logscatter22.png')

###Fig 4A (RIGHT) - Example refractory threshold for REMpre = 60s

#Calculate mean and st.dev for REMpre = 60s
rem=60
m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc

#Define CDF
fplot = lambda x: stats.norm.cdf(x,m1,s1)

#x-values to plot
xplot = np.arange(2,9,0.01)

fig4A2 = plt.figure()
plt.plot(fplot(xplot),xplot,color=col1)
plt.axvline(x=0.01, color='k', lw=1, ls='--')
plt.ylim([2,9])
plt.yticks([2,4,6,8],["","","",""])
plt.xticks([0,0.5,1],["","",""])
sns.despine()

fig4A2.savefig(outpath+'f4A_cdfex.pdf')


###############################################################################
###Fig 4B - Bootstrap results of refractory threshold

#Reset index of dataframe
remDF.reset_index(drop=True)

#Define log function
def log_func(x, a, b, c):
    return a*np.log(x+b)+c

#List to store refractory thresholds
bootstrap_thresholds = []

#Run gmm estimation and model fitting for 10,000 bootstrap iterations
for i in range(10000):
    np.random.seed(i)
    sample = np.random.choice(5090,5090)
    logsws30 = []
    logsws60 = []
    logsws90 = []
    logsws120 = []
    logsws150 = []
    logsws180 = []
    logsws210 = []
    logsws240 = []
    for x in sample:
        row = remDF.iloc[x]
        rem = row[0]
        logsws = row[9]
        if rem<30:
            logsws30.append(logsws)
        elif rem<60:
            logsws60.append(logsws)
        elif rem<90:
            logsws90.append(logsws)
        elif rem<120:
            logsws120.append(logsws)
        elif rem<150:
            logsws150.append(logsws)
        elif rem<180:
            logsws180.append(logsws)
        elif rem<210:
            logsws210.append(logsws)
        else:
            logsws240.append(logsws)
    logdatas = [logsws30,logsws60,logsws90,logsws120,logsws150,logsws180,logsws210,logsws240]
    khigh,klow,mhigh,mlow,shigh,slow = rp.gmm_params(logdatas)
    xes = [15,45,75,105,135,165,195,225]
    mhigh_log = curve_fit(log_func, xes, mhigh)[0]
    mhigh_loga = mhigh_log[0]
    mhigh_logb = mhigh_log[1]
    mhigh_logc = mhigh_log[2]
    shigh_log = curve_fit(log_func, xes, shigh, p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
    shigh_loga = shigh_log[0]
    shigh_logb = shigh_log[1]
    shigh_logc = shigh_log[2]
    upper1 = []
    rems = np.arange(7.5,242.5,2.5)
    for rem in rems:
        m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
        s1 = shigh_loga*np.log(rem + shigh_logb)+shigh_logc
        upper1.append(m1 - 2.326*s1)
    exp_upper1 = np.exp(upper1)
    bootstrap_thresholds.append(exp_upper1)
    print('iteration:'+str(i))

#x(REMpre) values to plot
rems = np.arange(7.5,242.5,2.5)

#Calculate 95% threshold
ubound = np.percentile(bootstrap_thresholds, 99.5, axis=0)
lbound = np.percentile(bootstrap_thresholds, 0.5, axis=0)
means = np.mean(bootstrap_thresholds, axis=0)


fig4B = plt.figure()
plt.scatter(sinDF.rem, sinDF.sws, s=10, color=col1, alpha=0.6)
plt.scatter(nonDF.rem, nonDF.sws, s=10, color='gray', alpha=0.6)
plt.scatter(seqDF.rem, seqDF.sws, s=10, color='gray', alpha=0.6)
plt.plot(rems,means, color='k')
plt.fill_between(rems,lbound,ubound,color='k', alpha=0.4)
sns.despine()
plt.yticks([0,500,1000,1500,2000],["","","","",""])
plt.xticks([0,60,120,180,240],["","","","",""])
plt.xlim([0,240])
plt.ylim([0,2010])

fig4B.savefig(outpath+'bootstrap_ref2.png')

###############################################################################
### Cycles that entered next REM during the refractory period

#Reset index for single cycle dataframe
sinDF = sinDF.reset_index(drop=True)

#Count number of single cycles that ended during refractory period
inPermissive = []
for idx,row in sinDF.iterrows():
    rem = row['rem']
    sws = row['sws']
    thresh = rp.athreshold(rem, thresh1)
    if thresh<sws:
        inPermissive.append(False)
    else:
        inPermissive.append(True)

#Calculate as proportion of all single cycles
sum(inPermissive)/len(sinDF)


###############################################################################
### Fig 4C,D,E - Spectral density, spindles, MAs comparison between Refractory
#and permissive period

#Find recordigns with Spindles
spindleRecs = []
for rec in recordings:
    if os.path.isfile(ppath+rec+'/'+'vip_'+rec+'.txt'):
        spindleRecs.append(rec)

#Lists to store spectral density, spindles, MAs
refSpec = []
permSpec = []
rSpec = []
pSpec = []


twin=3

#Loop through all single cycles in all recordings and calculate spectral density,
#spindles/min., MAs/min. for both refractory and permissive zones
for rec in spindleRecs:
    #Load recording and find cycles
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
        #Loop through all cycles
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240):
                if rp.isSequential(rem, sws, intersect_x, intersect_y)==False: #Check if single cycle
                    refraSeq, permSeq = rp.find_refractory(sub, thresh1)
                    if (len(refraSeq)>0)&(len(permSeq)>0):
                        if (len(refraSeq[0])>=2)&(len(permSeq[0])>=2):
                            subrefspec = []
                            subpermspec = []
                            #Calculate Spectral density
                            for s in refraSeq:
                                if len(s)*nbin>=nwin:
                                    b = int((s[-1]+1)*nbin)
                                    sup = list(range(int(s[0]*nbin), b))
                                    if sup[-1]>len(EEG):
                                        sup = list(range(int(s[0]*nbin), len(EEG)))
                                    if len(sup) >= nwin:
                                        Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                                        subrefspec.append(Pow)
                                        rSpec.append(Pow)
                            for s in permSeq:
                                if len(s)*nbin>=nwin:
                                    b = int((s[-1]+1)*nbin)
                                    sup = list(range(int(s[0]*nbin), b))
                                    if sup[-1]>len(EEG):
                                        sup = list(range(int(s[0]*nbin), len(EEG)))
                                    if len(sup) >= nwin:
                                        Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                                        subpermspec.append(Pow)
                                        pSpec.append(Pow)
                            refSpec.append(np.mean(subrefspec,axis=0))
                            permSpec.append(np.mean(subpermspec,axis=0))
            cnt2+=1
    print(rec)

#Limit interest of frequencies to [0,30]Hz
ifreq2 = np.where(F<=30)
F2 = F[ifreq2]                

refraS = np.array([x[ifreq2] for x in refSpec])
permS = np.array([x[ifreq2] for x in permSpec])


#Calculate 99% Confidence intervals
arefraS = refraS.mean(axis=0) - 2.576*refraS.std(axis=0)/np.sqrt(len(refraS))
brefraS = refraS.mean(axis=0) + 2.576*refraS.std(axis=0)/np.sqrt(len(refraS))
apermS = permS.mean(axis=0) - 2.576*permS.std(axis=0)/np.sqrt(len(permS))
bpermS = permS.mean(axis=0) + 2.576*permS.std(axis=0)/np.sqrt(len(permS))

###Stats###

eeg2Freq = []
eeg2wpval = []

for i in range(46):
    curfreqind = i
    refp = np.array([x[curfreqind] for x in refraS])
    permp = np.array([x[curfreqind] for x in permS])
    eeg2Freq.append(F[i])
    eeg2wpval.append(stats.ttest_ind(refp,permp,equal_var=False)[1])

eeg2_sig1 = []
eeg2_sig2 = []
eeg2_sig3 = []
for idx,x in enumerate(eeg2wpval):
    if x<0.05:
        eeg2_sig1.append(eeg2Freq[idx])
    if x<0.01:
        eeg2_sig2.append(eeg2Freq[idx])
    if x<0.001:
        eeg2_sig3.append(eeg2Freq[idx])

eeg2_sigf1 = np.arange(0,15,0.01)


#Plot Spectral density
fig4C = plt.figure()
plt.plot(F2, refraS.mean(axis=0), label='Refractory', color='orange')
plt.plot(F2, permS.mean(axis=0), label='Permissive', color=(0.6, 0.8, 0.19))
plt.fill_between(F2, arefraS, brefraS, color='orange',alpha=0.5)
plt.fill_between(F2, apermS, bpermS, color=(0.6, 0.8, 0.19), alpha=0.5)
plt.plot(eeg2_sigf1, np.repeat(1100,len(eeg2_sigf1)),color='k',lw=1)
plt.legend()
sns.despine()
plt.ylim([0,1200])
plt.yticks([0,400,800,1200])

fig4C.savefig(outpath+'f4c_spdensity.pdf')


###############################################################################
### Fig 4D,E - Spindles, MAs comparison between Refractory and permissive period


refMAs = []
permMAs = []
refSpindles = []
permSpindles = []

#Loop through all single cycles in all recordings and calculate
#spindles/min., MAs/min. for both refractory and permissive zones
for rec in spindleRecs:
    #Find spindle locations
    timestamp = []
    spindle = []
    with open(ppath + rec+'/'+'vip_'+rec+'.txt','r') as f:
        for line in f:
            if (line=='\n')or(line.startswith('@')):
                pass
            else:
                res = line.split('\t',-1)
                timestamp.append(res[0])
                if res[1]=='':
                    spindle.append(0)
                elif res[1]=='o':
                    spindle.append(0)
                else:
                    spindle.append(1)
    f.close()   
    spindleloc = np.where(np.array(spindle)==1)[0]
    spindleranges = rp.ranges(spindleloc)
    scaledSpindleTimes = []
    for x in spindleranges:
        startind = x[0]
        endind = x[1]
        starter = float(timestamp[startind])
        ender = float(timestamp[endind])
        indstart = starter/2.5
        indend = ender/2.5
        scaledSpindleTimes.append((indstart, indend))
    #Load recording and find cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    M,S = rp.load_stateidx(ppath, rec)
    revvec = rp.nts(M)
    revtup = rp.vecToTup(revvec)
    revfixtup = rp.ma_thr(revtup, 8)
    fixvec = rp.tupToVec(revfixtup)
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
        #Loop through all cycles
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240):
                if rp.isSequential(rem, sws, intersect_x, intersect_y)==False: #Check if single cycle
                    refraSeq, permSeq = rp.find_refractory(sub, thresh1)
                    if len(refraSeq)>0:
                        #Count Spindles and MAs
                        seqstart = refraSeq[0][0]
                        seqend = refraSeq[-1][-1]
                        subfixvec1 = fixvec[seqstart:seqend+1]
                        subtup1 = rp.vecToTup(subfixvec1)
                        seq2start = permSeq[0][0]
                        seq2end = permSeq[-1][-1]
                        subfixvec2 = fixvec[seq2start:seq2end+1]
                        subtup2 = rp.vecToTup(subfixvec2)
                        macnt1 = 0
                        macnt2 = 0
                        spcnt1 = 0
                        spcnt2 = 0
                        refsws = 0
                        permsws = 0
                        for y in subtup1:
                            if y[0]=='MA':
                                macnt1+=1
                            if (y[0]=='N')or(y[0]=='MA'):
                                refsws+=y[1]*2.5
                        for y in subtup2:
                            if y[0]=='MA':
                                macnt2+=1
                            if (y[0]=='N')or(y[0]=='MA'):
                                permsws+=y[1]*2.5
                        for y in scaledSpindleTimes:
                            spinstart = y[0]
                            spinend = y[1]
                            if (spinstart>=seqstart)&(spinend<=seqend):
                                spcnt1+=1
                            if (spinstart>=seq2start)&(spinend<=seq2end):
                                spcnt2+=1
                        refmarate = macnt1/refsws*60
                        permmarate = macnt2/permsws*60
                        refsprate = spcnt1/refsws*60
                        permsprate = spcnt2/permsws*60
                        #Add results to lists
                        refMAs.append(refmarate)                                       
                        permMAs.append(permmarate)
                        refSpindles.append(refsprate)
                        permSpindles.append(permsprate)
            cnt2+=1
    print(rec)


###Stats###

##Spindles
stats.levene(refSpindles, permSpindles)
stats.ttest_ind(refSpindles,permSpindles,equal_var=False)

##MAs
stats.levene(refMAs,permMAs)
stats.ttest_ind(refMAs,permMAs,equal_var=False)


#Plot boxplots of Spindles/min.
boxlabels = ['Refractory','Permissive']
boxcolors = ['orange',(0.6, 0.8, 0.19)]

fig5,ax = plt.subplots()
bp = ax.boxplot([refSpindles, permSpindles], vert=True, patch_artist=True,labels=boxlabels)
for ind,patch in enumerate(bp['boxes']):
    patch.set_facecolor(boxcolors[ind])
sns.despine()

#Plot boxplots of MAs/min.
fig6,ax = plt.subplots()
bp = ax.boxplot([refMAs, permMAs], vert=True, patch_artist=True, labels=boxlabels)
for ind,patch in enumerate(bp['boxes']):
    patch.set_facecolor(boxcolors[ind])
sns.despine()


fig5.savefig(outpath+'f4_refpermspin.pdf')
fig6.savefig(outpath+'f4_refpermma.pdf')

###############################################################################
### Fig 4F - Progression of delta,theta,sigma,spindles,MAs

#Lists to store results for different lengths of REMpre
spindles60 = []
spindles120 = []
spindles180 = []
spindles240 = []

mas60 = []
mas120 = []
mas180 = []
mas240 = []

thetapows60 = []
thetapows120 = []
thetapows180 = []
thetapows240 = []

sigmapows60 = []
sigmapows120 = []
sigmapows180 = []
sigmapows240 = []

twin=3

#Loop through all single cycles in all recordings and calculate the progression
#of sigma,theta,delta,spindles,MAs through quarters of Refractory and quarters
#of permissive period. Do this for different lengths of REMpre.

for rec in spindleRecs:
    #Find spindle locations
    timestamp = []
    spindle = []
    
    with open(ppath + rec+'/'+'vip_'+rec+'.txt','r') as f:
        for line in f:
            if (line=='\n')or(line.startswith('@')):
                pass
            else:
                res = line.split('\t',-1)
                timestamp.append(res[0])
                if res[1]=='':
                    spindle.append(0)
                elif res[1]=='o':
                    spindle.append(0)
                else:
                    spindle.append(1)
    f.close()   
    spindleloc = np.where(np.array(spindle)==1)[0]
    spindleranges = rp.ranges(spindleloc)
    scaledSpindleTimes = []
    for x in spindleranges:
        startind = x[0]
        endind = x[1]
        starter = float(timestamp[startind])
        ender = float(timestamp[endind])
        indstart = starter/2.5
        indend = ender/2.5
        scaledSpindleTimes.append((indstart, indend))
    #Load recording and find cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    M,S = rp.load_stateidx(ppath, rec)
    revvec = rp.nts(M)
    revtup = rp.vecToTup(revvec)
    revfixtup = rp.ma_thr(revtup, 8)
    fixvec = rp.tupToVec(revfixtup)
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
        #Loop through all cycles
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240):
                if rp.isSequential(rem,sws,intersect_x,intersect_y)==False:#Check if single cycle
                    refQ = rp.refra_quarters(sub, thresh1)
                    permQ = rp.perm_quarters(sub, thresh1)
                    if (len(refQ[0])>0)&(len(permQ[0])>0):
                        allQ = []
                        allQ.extend(refQ)
                        allQ.extend(permQ)
                        sigmapows = []
                        thetapows = []
                        sprates = []
                        spcounts = []
                        marates = []
                        macounts = []
                        #Count/Calculate all values
                        for quarter in allQ:
                            #MA
                            seqstart = quarter[0][0]
                            seqend = quarter[-1][-1]
                            subfixvec = fixvec[seqstart:seqend+1]
                            subtup = rp.vecToTup(subfixvec)
                            macnt = 0
                            for y in subtup:
                                if y[0]=='MA':
                                    macnt+=1
                            macounts.append(macnt)
                            #Spindle
                            spcount = 0
                            for y in scaledSpindleTimes:
                                spinstart = y[0]
                                spinend = y[1]
                                if (spinstart>=seqstart)&(spinend<=seqend):
                                    spcount+=1
                            spcounts.append(spcount)
                            subsigmapows = []
                            subthetapows = []
                            #Sigma,Theta
                            for s in quarter:
                                if len(s)*nbin>=nwin:
                                    b = int((s[-1]+1)*nbin)
                                    sup = list(range(int(s[0]*nbin),b))
                                    if sup[-1]>len(EEG):
                                        sup = list(range(int(s[0]*nbin), len(EEG)))
                                    if len(sup) >= nwin:
                                        Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                        df = F[1]-F[0]
                                        thetafreq = np.where((F>=5)&(F<=9.5))
                                        sigmafreq = np.where((F>=10)&(F<=15))
                                        sigmapow = np.sum(Pow[sigmafreq])*df
                                        thetapow = np.sum(Pow[thetafreq])*df
                                        subthetapows.append(thetapow)
                                        subsigmapows.append(sigmapow)
                                    else:
                                        subsigmapows.append(np.nan)
                                        subthetapows.append(np.nan)
                                else:
                                    subsigmapows.append(np.nan)
                                    subthetapows.append(np.nan)
                            thetapows.append(np.nanmean(subthetapows))
                            sigmapows.append(np.nanmean(subsigmapows))
                        #Account for overlaps in MAs and Spindles
                        for idx,quarter in enumerate(allQ):
                            if idx<7:
                                seqstart = quarter[0][0]
                                seqend = quarter[-1][-1]
                                quarter2 = allQ[idx+1]
                                seq2start = quarter2[0][0]
                                seq2end = quarter2[-1][-1]
                                subfixvec1 = fixvec[seqstart:seqend+1]
                                subfixvec2 = fixvec[seq2start:seq2end+1]
                                subtup1 = rp.vecToTup(subfixvec1)
                                subtup2 = rp.vecToTup(subfixvec2)
                                if (subtup1[-1][0]=='MA')&(subtup2[0][0]=='MA'):
                                    if subtup1[-1][1]>=subtup2[0][1]:
                                        macounts[idx+1] = macounts[idx+1]-1
                                    else:
                                        macounts[idx] = macounts[idx]-1
                                for y in scaledSpindleTimes:
                                    spinstart = y[0]
                                    spinend = y[1]
                                    if (spinstart<=seqend)&(spinend>=seq2start):
                                        if (seqend-spinstart)>=(spinend-seq2start):
                                            spcounts[idx] = spcounts[idx]+1
                                        else:
                                            spcounts[idx+1] = spcounts[idx+1]+1
                        #Convert count to frequency (count/min.)
                        for idx,quarter in enumerate(allQ):
                            seqstart = quarter[0][0]
                            seqend = quarter[-1][-1]
                            subfixvec = fixvec[seqstart:seqend+1]
                            subtup = rp.vecToTup(subfixvec)
                            nremamt = 0
                            for y in subtup:
                                if (y[0]=='N')or(y[0]=='MA'):
                                    nremamt+=y[1]*2.5
                            macnt = macounts[idx]
                            spcnt = spcounts[idx]
                            sprates.append(spcnt/nremamt*60)
                            marates.append(macnt/nremamt*60)
                        mname = rec.split('_')[0]
                        #Add results to lists
                        if rem<60:
                            thetapows60.append(thetapows)
                            sigmapows60.append(sigmapows)
                            spindles60.append(sprates)
                            mas60.append(marates)
                        elif rem<120:
                            thetapows120.append(thetapows)
                            sigmapows120.append(sigmapows)
                            spindles120.append(sprates)
                            mas120.append(marates)       
                        elif rem<180:
                            thetapows180.append(thetapows)
                            sigmapows180.append(sigmapows)
                            spindles180.append(sprates)
                            mas180.append(marates)
                        else:
                            thetapows240.append(thetapows)
                            sigmapows240.append(sigmapows)
                            spindles240.append(sprates)
                            mas240.append(marates)
            cnt2+=1
    print(rec)

allThetas = [thetapows60,thetapows120,thetapows180,thetapows240]
allSigmas = [sigmapows60,sigmapows120,sigmapows180,sigmapows240]
allSpindles = [spindles60,spindles120,spindles180,spindles240]
allMAs = [mas60,mas120,mas180,mas240]

#Compile results regardless of REMpre
sigscomp = []
for x in allSigmas:
    sigscomp.extend(x)
thescomp = []
for x in allThetas:
    thescomp.extend(x)
spinscomp = []
for x in allSpindles:
    spinscomp.extend(x)
mascomp = []
for x in allMAs:
    mascomp.extend(x)

#xvalue to plot
xes = np.arange(0.5,8,1)


##Theta Power

#99% Conf. Interval
aThe = np.nanmean(thescomp,axis=0) - 2.576*np.nanstd(thescomp,axis=0)/np.sqrt(len(thescomp))
bThe = np.nanmean(thescomp,axis=0) + 2.576*np.nanstd(thescomp,axis=0)/np.sqrt(len(thescomp))

#Plot
fig4F2 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,theta in enumerate(allThetas):
    if idx==0:
        low=7.5
        high=60
    else:
        low=60*idx
        high=60*(idx+1)
    lbl = str(low)+'<=rem<'+str(high)
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(theta,axis=0), color=colorVal, label=lbl)
    plt.plot(xes, np.nanmean(theta,axis=0), color=colorVal)
plt.scatter(xes, np.nanmean(thescomp,axis=0), color='k')
plt.plot(xes, np.nanmean(thescomp,axis=0), color='k')    
plt.fill_between(xes, aThe, bThe, color='k', alpha=0.2)
plt.axvline(x=4, color='k', lw=0.5, ls='--')
sns.despine()
plt.legend()

fig4F2.savefig(outpath+'f4f_eeg2theta2.pdf')


##Sigma Power

#99% Conf. Interval
aSig = np.nanmean(sigscomp,axis=0) - 2.576*np.nanstd(sigscomp,axis=0)/np.sqrt(len(sigscomp))
bSig = np.nanmean(sigscomp,axis=0) + 2.576*np.nanstd(sigscomp,axis=0)/np.sqrt(len(sigscomp))

#Plot
fig4F3 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,sigma in enumerate(allSigmas):
    if idx==0:
        low=7.5
        high=60
    else:
        low=60*idx
        high=60*(idx+1)
    lbl = str(low)+'<=rem<'+str(high)
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(sigma,axis=0), color=colorVal, label=lbl)
    plt.plot(xes, np.nanmean(sigma,axis=0), color=colorVal)
plt.scatter(xes, np.nanmean(sigscomp,axis=0), color='k')
plt.plot(xes, np.nanmean(sigscomp,axis=0), color='k')
plt.fill_between(xes, aSig, bSig, color='k', alpha=0.2)
plt.axvline(x=4, color='k', lw=0.5, ls='--')
sns.despine()
plt.legend()

fig4F3.savefig(outpath+'f4f_eeg2sigma2.pdf')


##Spindles/min.

#99% Conf. Interval
aSpin = np.nanmean(spinscomp,axis=0) - 2.576*np.nanstd(spinscomp,axis=0)/np.sqrt(len(spinscomp))
bSpin = np.nanmean(spinscomp,axis=0) + 2.576*np.nanstd(spinscomp,axis=0)/np.sqrt(len(spinscomp))

#Plot
fig4G1 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,delsig in enumerate(allSpindles):
    if idx==0:
        low=7.5
        high=60
    else:
        low=60*idx
        high=60*(idx+1)
    lbl = str(low)+'<=rem<'+str(high)
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(delsig,axis=0), color=colorVal, label=lbl)
    plt.plot(xes, np.nanmean(delsig,axis=0), color=colorVal)
plt.scatter(xes, np.nanmean(spinscomp,axis=0), color='k')
plt.plot(xes, np.nanmean(spinscomp,axis=0), color='k')
plt.fill_between(xes, aSpin, bSpin, color='k', alpha=0.2)
plt.axvline(x=4, color='k', lw=0.5, ls='--')
sns.despine()
plt.legend()

fig4G1.savefig(outpath+'f4G_rel_spindles2.pdf')


##MAs/min.

#99% Conf. Interval
aMA = np.nanmean(mascomp,axis=0) - 2.576*np.nanstd(mascomp,axis=0)/np.sqrt(len(mascomp))
bMA = np.nanmean(mascomp,axis=0) + 2.576*np.nanstd(mascomp,axis=0)/np.sqrt(len(mascomp))

#Plot
fig4G2 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,delsig in enumerate(allMAs):
    if idx==0:
        low=7.5
        high=60
    else:
        low=60*idx
        high=60*(idx+1)
    lbl = str(low)+'<=rem<'+str(high)
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(delsig,axis=0), color=colorVal, label=lbl)
    plt.plot(xes, np.nanmean(delsig,axis=0), color=colorVal)
plt.scatter(xes, np.nanmean(mascomp,axis=0), color='k')
plt.plot(xes, np.nanmean(mascomp,axis=0), color='k')
plt.fill_between(xes, aMA, bMA, color='k', alpha=0.2)
plt.axvline(x=4, color='k', lw=0.5, ls='--')
sns.despine()
plt.legend()

fig4G2.savefig(outpath+'f4g_rel_ma2.pdf')



###############################################################################
### Fig 4H - Progression of Sigma power in absolute time

#List to store sigma power by different REMpre values
absSigmas60 = []
absSigmas120 = []
absSigmas180 = []
absSigmas240 = []

absThetas60 = []
absThetas120 = []
absThetas180 = []
absThetas240 = []

twin=3

for rec in spindleRecs:
    #Load recordings and define cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    M,S = rp.load_stateidx(ppath, rec)
    revvec = rp.nts(M)
    revtup = rp.vecToTup(revvec)
    revfixtup = rp.ma_thr(revtup, 8)
    fixvec = rp.tupToVec(revfixtup)    
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
        #Loop through all cycles
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            wake=0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
                if (x[0]=='W'):
                    wake+=x[1]*2.5
            inter = sws+wake
            if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240]
                if rp.isSequential(rem,sws,intersect_x,intersect_y)==False: #Check if cycle is single
                    #Find |N| sequences (30 seconds per bin)
                    nowakesub = [x for x in sub if x[0]!='W']
                    totlist = []
                    for x in nowakesub[1:]:
                        totlist.extend(list(np.arange(x[2]-x[1]+1,x[2]+1)))
                    allSeqs = []
                    for i in range(34):
                        if len(totlist)>12*(i+1):
                            allSeqs.append(totlist[12*i:12*(i+1)])
                        else:
                            allSeqs.append(None)
                    allRanges = []
                    for seq in allSeqs:
                        if seq!=None:
                            subranges = rp.ranges(seq)
                            allRanges.append(subranges)
                        else:
                            allRanges.append([])
                    #Calculate sigma powers
                    allThetas = []
                    allSigmas = []
                    for subranges in allRanges:
                        if len(subranges)>0:
                            subsigmas = []
                            subthetas = []
                            for s in subranges:
                                b = int((s[-1]+1)*nbin)
                                sup = list(range(int(s[0]*nbin),b))
                                if len(sup)>=nwin:
                                    Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                    df = F[1]-F[0]
                                    sigmafreq = np.where((F>=10)&(F<=15))
                                    thetafreq = np.where((F>=5)&(F<=9.5))
                                    sigmapow = np.sum(Pow[sigmafreq])*df
                                    thetapow = np.sum(Pow[thetafreq])*df
                                    subsigmas.append(sigmapow)
                                    subthetas.append(thetapow)
                                else:
                                    subsigmas.append(np.nan)
                                    subthetas.append(np.nan)
                            allSigmas.append(np.nanmean(subsigmas))
                            allThetas.append(np.nanmean(subthetas))
                        else:
                            allSigmas.append(np.nan)
                            allThetas.append(np.nan)
                    if rem<60:
                        absSigmas60.append(allSigmas)
                        absThetas60.append(allThetas)
                    elif rem<120:
                        absSigmas120.append(allSigmas)
                        absThetas120.append(allThetas)
                    elif rem<180:
                        absSigmas180.append(allSigmas)
                        absThetas180.append(allThetas)
                    else:
                        absSigmas240.append(allSigmas)
                        absThetas240.append(allThetas)
            cnt2+=1
    print(rec)

#xvalues to plot
xes = np.arange(15,1015,30)

### Sigma

##REMpre = 60s

#99% Conf. Interval
asig60 = np.nanmean(absSigmas60,axis=0) - 2.576*np.nanstd(absSigmas60,axis=0)/np.sqrt(len(absSigmas60))
bsig60 = np.nanmean(absSigmas60,axis=0) + 2.576*np.nanstd(absSigmas60,axis=0)/np.sqrt(len(absSigmas60))

#Refractory threshold
lowthre60 = rp.athreshold(7.5, thresh1)
highthre60 = rp.athreshold(60,thresh1)

#Plot
fig4H1 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(absSigmas60,axis=0), color=col)
plt.fill_between(xes, asig60,bsig60, color=col, alpha=0.4)
plt.axvline(x=lowthre60, color='k', lw=1, ls='--')
plt.axvline(x=highthre60, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

fig4H1.savefig(outpath+'f4h_abssig60.pdf')


##REMpre = 120s

#99% Conf. Interval
asig120 = np.nanmean(absSigmas120,axis=0) - 2.576*np.nanstd(absSigmas120,axis=0)/np.sqrt(len(absSigmas120))
bsig120 = np.nanmean(absSigmas120,axis=0) + 2.576*np.nanstd(absSigmas120,axis=0)/np.sqrt(len(absSigmas120))

#Refractory threshold
lowthre120 = rp.athreshold(60, thresh1)
highthre120 = rp.athreshold(120,thresh1)

#Plot
fig4H2 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(absSigmas120,axis=0), color=col)
plt.fill_between(xes, asig120,bsig120, color=col, alpha=0.4)
plt.axvline(x=lowthre120, color='k', lw=1, ls='--')
plt.axvline(x=highthre120, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

fig4H2.savefig(outpath+'f4h_abssig120.pdf')


##REMpre = 180s

#99% Conf. Interval
asig180 = np.nanmean(absSigmas180,axis=0) - 2.576*np.nanstd(absSigmas180,axis=0)/np.sqrt(len(absSigmas180))
bsig180 = np.nanmean(absSigmas180,axis=0) + 2.576*np.nanstd(absSigmas180,axis=0)/np.sqrt(len(absSigmas180))

#Refractory threshold
lowthre180 = rp.athreshold(120, thresh1)
highthre180 = rp.athreshold(180,thresh1)

#Plot
fig4H3 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(absSigmas180,axis=0), color=col)
plt.fill_between(xes, asig180,bsig180, color=col, alpha=0.4)
plt.axvline(x=lowthre180, color='k', lw=1, ls='--')
plt.axvline(x=highthre180, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

fig4H3.savefig(outpath+'f4h_abssig180.pdf')


##REMpre = 240s

#99% Conf. Interval
asig240 = np.nanmean(absSigmas240,axis=0) - 2.576*np.nanstd(absSigmas240,axis=0)/np.sqrt(len(absSigmas240))
bsig240 = np.nanmean(absSigmas240,axis=0) + 2.576*np.nanstd(absSigmas240,axis=0)/np.sqrt(len(absSigmas240))

#Refractory threshold
lowthre240 = rp.athreshold(180, thresh1)
highthre240 = rp.athreshold(240,thresh1)

#Plot
fig4H4 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(absSigmas240,axis=0), color=col)
plt.fill_between(xes, asig240,bsig240, color=col, alpha=0.4)
plt.axvline(x=lowthre240, color='k', lw=1, ls='--')
plt.axvline(x=highthre240, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

fig4H4.savefig(outpath+'f4H_abssig240.pdf')


### Theta

#REMpre = 60
athe60 = np.nanmean(absThetas60,axis=0) - 2.576*np.nanstd(absThetas60,axis=0)/np.sqrt(len(absThetas60))
bthe60 = np.nanmean(absThetas60,axis=0) + 2.576*np.nanstd(absThetas60,axis=0)/np.sqrt(len(absThetas60))

fig4H5 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(absThetas60,axis=0), color=col)
plt.fill_between(xes, athe60, bthe60, color=col, alpha=0.4)
plt.axvline(x=lowthre60, color='k', lw=1, ls='--')
plt.axvline(x=highthre60, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])

#REMpre = 120
athe120 = np.nanmean(absThetas120,axis=0) - 2.576*np.nanstd(absThetas120,axis=0)/np.sqrt(len(absThetas120))
bthe120 = np.nanmean(absThetas120,axis=0) + 2.576*np.nanstd(absThetas120,axis=0)/np.sqrt(len(absThetas120))

fig4H6 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(absThetas120,axis=0), color=col)
plt.fill_between(xes, athe120, bthe120, color=col, alpha=0.4)
plt.axvline(x=lowthre120, color='k', lw=1, ls='--')
plt.axvline(x=highthre120, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])

#REMpre = 180
athe180 = np.nanmean(absThetas180,axis=0) - 2.576*np.nanstd(absThetas180,axis=0)/np.sqrt(len(absThetas180))
bthe180 = np.nanmean(absThetas180,axis=0) + 2.576*np.nanstd(absThetas180,axis=0)/np.sqrt(len(absThetas180))

fig4H7 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(absThetas180,axis=0), color=col)
plt.fill_between(xes, athe180, bthe180, color=col, alpha=0.4)
plt.axvline(x=lowthre180, color='k', lw=1, ls='--')
plt.axvline(x=highthre180, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])

#REMpre = 240
athe240 = np.nanmean(absThetas240,axis=0) - 2.576*np.nanstd(absThetas240,axis=0)/np.sqrt(len(absThetas240))
bthe240 = np.nanmean(absThetas240,axis=0) + 2.576*np.nanstd(absThetas240,axis=0)/np.sqrt(len(absThetas240))

fig4H8 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(absThetas240,axis=0), color=col)
plt.fill_between(xes, athe240, bthe240, color=col, alpha=0.4)
plt.axvline(x=lowthre240, color='k', lw=1, ls='--')
plt.axvline(x=highthre240, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])

fig4H5.savefig(outpath+'f4_abstheta60.pdf')
fig4H6.savefig(outpath+'f4_abstheta120.pdf')
fig4H7.savefig(outpath+'f4_abstheta180.pdf')
fig4H8.savefig(outpath+'f4_abstheta240.pdf')

