#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:01:53 2021

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
from scipy.optimize import fsolve
from scipy import stats
import scipy.io as so
import matplotlib.colors as colors
import matplotlib.cm as cmx

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/REM_GMM/final_figures/Sfig4/'

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
### A - Spectral density

#Lists to store spectral density
refSpec = []
permSpec = []

twin=3

#Loop through all single cycles in all recordings and calculate spectral density,
#spindles/min., MAs/min. for both refractory and permissive zones
for rec in recordings:
    #Load recording and find cycles
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
                            for s in permSeq:
                                if len(s)*nbin>=nwin:
                                    b = int((s[-1]+1)*nbin)
                                    sup = list(range(int(s[0]*nbin), b))
                                    if sup[-1]>len(EEG):
                                        sup = list(range(int(s[0]*nbin), len(EEG)))
                                    if len(sup) >= nwin:
                                        Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                                        subpermspec.append(Pow)
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

##Spectral density
eeg2Freq = []
eeg2wpval = []

for i in range(46):
    curfreqind = i
    refp = np.array([x[curfreqind] for x in refSpec])
    permp = np.array([x[curfreqind] for x in permSpec])
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

#5%
eeg2_sig1f1 = np.arange(0,4.6666,0.0001)
eeg2_sig1f2 = np.arange(6.3333,15,0.0001)
#1%
eeg2_sig2f1 = np.arange(0,4.6666,0.0001)
eeg2_sig2f2 = np.arange(6.3333,15,0.0001)
#0.1%
eeg2_sig3f1 = np.arange(0.6666,4.3333,0.0001)
eeg2_sig3f2 = np.arange(6.3333,15,0.0001)

#Plot Spectral density
sfig3A = plt.figure()
plt.plot(F2, refraS.mean(axis=0), label='Refractory', color='orange')
plt.plot(F2, permS.mean(axis=0), label='Permissive', color=(0.6, 0.8, 0.19))
plt.fill_between(F2, arefraS, brefraS, color='orange',alpha=0.5)
plt.fill_between(F2, apermS, bpermS, color=(0.6, 0.8, 0.19), alpha=0.5)
plt.plot(eeg2_sig1f1, np.repeat(1050,len(eeg2_sig1f1)),color='k',lw=1)
plt.plot(eeg2_sig1f2, np.repeat(1050,len(eeg2_sig1f2)),color='k',lw=1)
plt.plot(eeg2_sig2f1, np.repeat(1150,len(eeg2_sig2f1)),color='k',lw=1)
plt.plot(eeg2_sig2f2, np.repeat(1150,len(eeg2_sig2f2)),color='k',lw=1)
plt.plot(eeg2_sig3f1, np.repeat(1250,len(eeg2_sig3f1)),color='k',lw=1)
plt.plot(eeg2_sig3f2, np.repeat(1250,len(eeg2_sig3f2)),color='k',lw=1)
plt.legend()
sns.despine()
plt.ylim([0,1300])
plt.yticks([0,400,800,1200])

sfig3A.savefig(outpath+'sf3a_spectra.pdf')


###############################################################################
### B - prefrontal delta progression

spindleRecs = []
for rec in recordings:
    if os.path.isfile(ppath+rec+'/'+'vip_'+rec+'.txt'):
        spindleRecs.append(rec)

deltapows60 = []
deltapows120 = []
deltapows180 = []
deltapows240 = []

twin=3

for rec in spindleRecs:
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
                        deltapows = []
                        #Count/Calculate all values
                        for quarter in allQ:
                            subdeltapows = []
                            for s in quarter:
                                if len(s)*nbin>=nwin:
                                    b = int((s[-1]+1)*nbin)
                                    sup = list(range(int(s[0]*nbin),b))
                                    if sup[-1]>len(EEG):
                                        sup = list(range(int(s[0]*nbin), len(EEG)))
                                    if len(sup) >= nwin:
                                        Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                        df = F[1]-F[0]
                                        deltafreq = np.where((F>=0.5)&(F<=4.5))
                                        deltapow = np.sum(Pow[deltafreq])*df
                                        subdeltapows.append(deltapow)
                                    else:
                                        subdeltapows.append(np.nan)
                                else:
                                    subdeltapows.append(np.nan)
                            deltapows.append(np.nanmean(subdeltapows))
                        #Add results to lists
                        if rem<60:
                            deltapows60.append(deltapows)
                        elif rem<120:
                            deltapows120.append(deltapows)
                        elif rem<180:
                            deltapows180.append(deltapows)
                        else:
                            deltapows240.append(deltapows)
            cnt2+=1
    print(rec)


allDeltas = [deltapows60,deltapows120,deltapows180,deltapows240]

delscomp = []
for x in allDeltas:
    delscomp.extend(x)

#99% Conf. Interval
aDel = np.nanmean(delscomp,axis=0) - 2.576*np.nanstd(delscomp,axis=0)/np.sqrt(len(delscomp))
bDel = np.nanmean(delscomp,axis=0) + 2.576*np.nanstd(delscomp,axis=0)/np.sqrt(len(delscomp))

#xvalue to plot
xes = np.arange(0.5,8,1)

#Plot
fig4F2 = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,delta in enumerate(allDeltas):
    if idx==0:
        low=7.5
        high=60
    else:
        low=60*idx
        high=60*(idx+1)
    lbl = str(low)+'<=rem<'+str(high)
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(delta,axis=0), color=colorVal, label=lbl)
    plt.plot(xes, np.nanmean(delta,axis=0), color=colorVal)
plt.scatter(xes, np.nanmean(delscomp,axis=0), color='k')
plt.plot(xes, np.nanmean(delscomp,axis=0), color='k')    
plt.fill_between(xes, aDel, bDel, color='k', alpha=0.2)
plt.axvline(x=4, color='k', lw=0.5, ls='--')
sns.despine()
plt.legend()


###############################################################################
### C - exclude last 40 s from permissive period
spindles60e = []
spindles120e = []
spindles180e = []
spindles240e = []

mas60e = []
mas120e = []
mas180e = []
mas240e = []

thetapows60e = []
thetapows120e = []
thetapows180e = []
thetapows240e = []

sigmapows60e = []
sigmapows120e = []
sigmapows180e = []
sigmapows240e = []

twin=3

#Find recordings with spindles
spindleRecs = []
for rec in recordings:
    if os.path.isfile(ppath+rec+'/'+'vip_'+rec+'.txt'):
        spindleRecs.append(rec)


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
                    allPerms = []
                    for x in permQ:
                        for y in x:
                            allPerms.extend(y)
                    if (len(refQ[0])>0)&(len(allPerms)>24):
                        allPerms2 = allPerms[:len(allPerms)-16]
                        splitlen = int(len(allPerms2)/4)
                        perm1 = allPerms2[0:splitlen]
                        perm2 = allPerms2[splitlen:2*splitlen]
                        perm3 = allPerms2[2*splitlen:3*splitlen]
                        perm4 = allPerms2[3*splitlen:]
                        perm1ranges = rp.ranges(perm1)
                        perm2ranges = rp.ranges(perm2)
                        perm3ranges = rp.ranges(perm3)
                        perm4ranges = rp.ranges(perm4)
                        perm1seq = []
                        perm2seq = []
                        perm3seq = []
                        perm4seq = []
                        for x in perm1ranges:
                            perm1seq.append(np.arange(x[0],x[1]+1))
                        for x in perm2ranges:
                            perm2seq.append(np.arange(x[0],x[1]+1))
                        for x in perm3ranges:
                            perm3seq.append(np.arange(x[0],x[1]+1))
                        for x in perm4ranges:
                            perm4seq.append(np.arange(x[0],x[1]+1))
                        permQ2 = []
                        permQ2.append(perm1seq)
                        permQ2.append(perm2seq)
                        permQ2.append(perm3seq)
                        permQ2.append(perm4seq)
                        allQ = []
                        allQ.extend(refQ)
                        allQ.extend(permQ2)
                        #
                        thetapows = []
                        sigmapows = []
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
                            #Sigma,Theta
                            subsigmapows = []
                            subthetapows = []
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
                            thetapows60e.append(thetapows)
                            sigmapows60e.append(sigmapows)
                            spindles60e.append(sprates)
                            mas60e.append(marates)
                        elif rem<120:
                            thetapows120e.append(thetapows)
                            sigmapows120e.append(sigmapows)
                            spindles120e.append(sprates)
                            mas120e.append(marates)       
                        elif rem<180:
                            thetapows180e.append(thetapows)
                            sigmapows180e.append(sigmapows)
                            spindles180e.append(sprates)
                            mas180e.append(marates)
                        else:
                            thetapows240e.append(thetapows)
                            sigmapows240e.append(sigmapows)
                            spindles240e.append(sprates)
                            mas240e.append(marates)
            cnt2+=1
    print(rec)


allThetas = [thetapows60e,thetapows120e,thetapows180e,thetapows240e]
allSigmas = [sigmapows60e,sigmapows120e,sigmapows180e,sigmapows240e]
allMAs = [mas60e,mas120e,mas180e,mas240e]
allSpins = [spindles60e,spindles120e,spindles180e,spindles240e]

compSigs = []
compSigs.extend(sigmapows60e)
compSigs.extend(sigmapows120e)
compSigs.extend(sigmapows180e)
compSigs.extend(sigmapows240e)
acompSigs = np.nanmean(compSigs,axis=0) - 2.576*np.nanstd(compSigs,axis=0)/np.sqrt(len(compSigs))
bcompSigs = np.nanmean(compSigs,axis=0) + 2.576*np.nanstd(compSigs,axis=0)/np.sqrt(len(compSigs))

compThes = []
compThes.extend(thetapows60e)
compThes.extend(thetapows120e)
compThes.extend(thetapows180e)
compThes.extend(thetapows240e)
acompThes = np.nanmean(compThes,axis=0) - 2.576*np.nanstd(compThes,axis=0)/np.sqrt(len(compThes))
bcompThes = np.nanmean(compThes,axis=0) + 2.576*np.nanstd(compThes,axis=0)/np.sqrt(len(compThes))

compMAs = []
compMAs.extend(mas60e)
compMAs.extend(mas120e)
compMAs.extend(mas180e)
compMAs.extend(mas240e)
acompMAs = np.nanmean(compMAs,axis=0) - 2.576*np.nanstd(compMAs,axis=0)/np.sqrt(len(compMAs))
bcompMAs = np.nanmean(compMAs,axis=0) + 2.576*np.nanstd(compMAs,axis=0)/np.sqrt(len(compMAs))

compSpins = []
compSpins.extend(spindles60e)
compSpins.extend(spindles120e)
compSpins.extend(spindles180e)
compSpins.extend(spindles240e)
acompSpins = np.nanmean(compSpins,axis=0) - 2.576*np.nanstd(compSpins,axis=0)/np.sqrt(len(compSpins))
bcompSpins = np.nanmean(compSpins,axis=0) + 2.576*np.nanstd(compSpins,axis=0)/np.sqrt(len(compSpins))

xes = np.arange(0.5,8,1)

#Sigma
fig10a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,subpows in enumerate(allSigmas):
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(subpows,axis=0),color=colorVal)
    plt.plot(xes, np.nanmean(subpows,axis=0),color=colorVal)
plt.plot(xes, np.nanmean(compSigs,axis=0), color='k')
plt.scatter(xes, np.nanmean(compSigs,axis=0),color='k')
plt.fill_between(xes,acompSigs,bcompSigs, color='k', alpha=0.3)
sns.despine()
plt.axvline(x=4, color='k', lw=1, ls='--')

#Theta
fig10b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,subpows in enumerate(allThetas):
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(subpows,axis=0),color=colorVal)
    plt.plot(xes, np.nanmean(subpows,axis=0),color=colorVal)
plt.plot(xes, np.nanmean(compThes,axis=0), color='k')
plt.scatter(xes, np.nanmean(compThes,axis=0),color='k')
plt.fill_between(xes,acompThes,bcompThes, color='k', alpha=0.3)
sns.despine()
plt.axvline(x=4, color='k', lw=1, ls='--')

#MA
fig10c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,subpows in enumerate(allMAs):
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(subpows,axis=0),color=colorVal)
    plt.plot(xes, np.nanmean(subpows,axis=0),color=colorVal)
plt.plot(xes, np.nanmean(compMAs,axis=0), color='k')
plt.scatter(xes, np.nanmean(compMAs,axis=0),color='k')
plt.fill_between(xes,acompMAs,bcompMAs, color='k', alpha=0.3)
sns.despine()
plt.axvline(x=4, color='k', lw=1, ls='--')

#Spindle
fig10d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,subpows in enumerate(allSpins):
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.scatter(xes, np.nanmean(subpows,axis=0),color=colorVal)
    plt.plot(xes, np.nanmean(subpows,axis=0),color=colorVal)
plt.plot(xes, np.nanmean(compSpins,axis=0), color='k')
plt.scatter(xes, np.nanmean(compSpins,axis=0),color='k')
plt.fill_between(xes,acompSpins,bcompSpins, color='k', alpha=0.3)
sns.despine()
plt.axvline(x=4, color='k', lw=1, ls='--')

fig10a.savefig(outpath+'sigma_e40.pdf')
fig10b.savefig(outpath+'theta_e40.pdf')
fig10c.savefig(outpath+'mas_e40.pdf')
fig10d.savefig(outpath+'spins_e40.pdf')


###############################################################################
### D - Spindles,MAs absolute progression after REM

#Find recordings with spindles
spindleRecs = []
for rec in recordings:
    if os.path.isfile(ppath+rec+'/'+'vip_'+rec+'.txt'):
        spindleRecs.append(rec)

#Lists storing results by REMpre duration
absDeltas60 = []
absDeltas120 = []
absDeltas180 = []
absDeltas240 = []

absThetas60 = []
absThetas120 = []
absThetas180 = []
absThetas240 = []

absSpindles60 = []
absSpindles120 = []
absSpindles180 = []
absSpindles240 = []

absMAs60 = []
absMAs120 = []
absMAs180 = []
absMAs240 = []

twin=3

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
        #Loop through all cycles
        cnt2 = 0
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240]
                if rp.isSequential(rem,sws,intersect_x,intersect_y)==False: #Check if cycle is single
                    #Find |N| within cycles
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
                    #Calculate quantities
                    allThetas = []
                    allDeltas = []
                    allMAcnts = []
                    allSpincnts = []
                    allSWSamt = []
                    for subranges in allRanges:
                        if len(subranges)>0:
                            subdeltas = []
                            subthetas = []
                            subspins = []
                            subMAs = []
                            subspincnt = []
                            submacnt = []
                            subswsamt = []
                            for s in subranges:
                                #Calculate Delta, Theta powers
                                b = int((s[-1]+1)*nbin)
                                sup = list(range(int(s[0]*nbin),b))
                                if len(sup)>=nwin:
                                    Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                    df = F[1]-F[0]
                                    deltafreq = np.where((F>=0.5)&(F<=4.5))
                                    thetafreq = np.where((F>=5)&(F<=9.5))
                                    deltapow = np.sum(Pow[deltafreq])*df
                                    thetapow = np.sum(Pow[thetafreq])*df
                                    subdeltas.append(deltapow)
                                    subthetas.append(thetapow)
                                else:
                                    subdeltas.append(np.nan)
                                    subthetas.append(np.nan)
                                #Count MAs and Spindles
                                if len(s)>0:
                                    #MA
                                    seqstart = s[0]
                                    seqend = s[-1]
                                    subfixvec = fixvec[seqstart:seqend+1]
                                    subtup = rp.vecToTup(subfixvec)
                                    swsamt = 0
                                    macnt = 0
                                    for y in subtup:
                                        if y[0]=='MA':
                                            macnt+=1
                                            swsamt+=y[1]*2.5
                                        if y[0]=='N':
                                            swsamt+=y[1]*2.5
                                    subswsamt.append(swsamt)
                                    submacnt.append(macnt)
                                    #Spindles
                                    spcnt = 0
                                    for y in scaledSpindleTimes:
                                        spinstart = y[0]
                                        spinend = y[1]
                                        if (spinstart>=seqstart)&(spinend<=seqend):
                                            spcnt+=1
                                    subspincnt.append(spcnt)
                                else:
                                    submacnt.append(np.nan)
                                    subspincnt.append(np.nan)
                            allDeltas.append(np.nanmean(subdeltas))
                            allThetas.append(np.nanmean(subthetas))
                            allMAcnts.append(np.nansum(submacnt))
                            allSpincnts.append(np.nansum(subspincnt))
                            allSWSamt.append(sum(subswsamt))
                        else:
                            allDeltas.append(np.nan)
                            allThetas.append(np.nan)
                            allSpincnts.append(np.nan)
                            allMAcnts.append(np.nan)
                            allSWSamt.append(np.nan)
                    #Account for overlaps in Spindles and MAs
                    for ind,subranges in enumerate(allRanges):
                        if ind<len(allRanges)-1:
                            nxtsubranges = allRanges[ind+1]
                            if (len(subranges)>0)&(len(nxtsubranges)>0):
                                curlast = subranges[-1]
                                nxtfirst = nxtsubranges[0]
                                if nxtfirst[0]==curlast[-1]+1:
                                    seqstart = curlast[0]
                                    seqend = curlast[-1]
                                    seq2start = nxtfirst[0]
                                    seq2end = nxtfirst[-1]
                                    subfixvec1 = fixvec[seqstart:seqend+1]
                                    subfixvec2 = fixvec[seq2start:seq2end+1]
                                    subtup1 = rp.vecToTup(subfixvec1)
                                    subtup2 = rp.vecToTup(subfixvec2)
                                    if (subtup1[-1][0]=='MA')&(subtup2[0][0]=='MA'):
                                        if subtup1[-1][1]>=subtup2[0][1]:
                                            allMAcnts[ind+1] = allMAcnts[ind+1]-1
                                        else:
                                            allMAcnts[ind] = allMAcnts[ind]-1
                                    for y in scaledSpindleTimes:
                                        spinstart = y[0]
                                        spinend = y[1]
                                        if (spinstart<=seqend)&(spinend>=seq2start):
                                            if (seqend-spinstart)>=(spinend-seq2start):
                                                allSpincnts[ind] = allSpincnts[ind]+1
                                            else:
                                                allSpincnts[ind+1] = allSpincnts[ind+1]+1
                    #Change counts into frequencies (/min.)
                    allSpins = []
                    allMAs = []
                    for ind in range(len(allRanges)):
                        maamt = allMAcnts[ind]
                        spinamt = allSpincnts[ind]
                        if np.isnan(maamt):
                            allSpins.append(np.nan)
                            allMAs.append(np.nan)
                        else:
                            allSpins.append(spinamt/allSWSamt[ind]*60)
                            allMAs.append(maamt/allSWSamt[ind]*60)
                    #Add results to lists
                    if rem<60:
                        absDeltas60.append(allDeltas)
                        absThetas60.append(allThetas)
                        absSpindles60.append(allSpins)
                        absMAs60.append(allMAs)
                    elif rem<120:
                        absDeltas120.append(allDeltas)
                        absThetas120.append(allThetas)
                        absSpindles120.append(allSpins)
                        absMAs120.append(allMAs)
                    elif rem<180:
                        absDeltas180.append(allDeltas)
                        absThetas180.append(allThetas)
                        absSpindles180.append(allSpins)
                        absMAs180.append(allMAs)
                    else:
                        absDeltas240.append(allDeltas)
                        absThetas240.append(allThetas)
                        absSpindles240.append(allSpins)
                        absMAs240.append(allMAs)
            cnt2+=1
    print(rec)

#Calculate refractory thresholds for different REMpre
lowthre60 = rp.athreshold(7.5, thresh1)
highthre60 = rp.athreshold(60,thresh1)
lowthre120 = rp.athreshold(60, thresh1)
highthre120 = rp.athreshold(120, thresh1)
lowthre180 = rp.athreshold(120, thresh1)
highthre180 = rp.athreshold(180, thresh1)
lowthre240 = rp.athreshold(180, thresh1)
highthre240 = rp.athreshold(240, thresh1)

#xvalues for plot
xes = np.arange(15,1015,30)



### Spindles

## 7.5s <= REMpre < 60s

#99% Confidence Interval
aspins60 = np.nanmean(absSpindles60, axis=0) - 2.576*np.nanstd(absSpindles60, axis=0)/np.sqrt(len(absSpindles60))
bspins60 = np.nanmean(absSpindles60, axis=0) + 2.576*np.nanstd(absSpindles60, axis=0)/np.sqrt(len(absSpindles60))

#Plot
fig3a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(absSpindles60,axis=0), color=col)
plt.fill_between(xes, aspins60,bspins60, color=col, alpha=0.4)
plt.axvline(x=lowthre60, color='k', lw=1, ls='--')
plt.axvline(x=highthre60, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## 60s <= REMpre < 120s

#99% Confidence Interval
aspins120 = np.nanmean(absSpindles120, axis=0) - 2.576*np.nanstd(absSpindles120, axis=0)/np.sqrt(len(absSpindles120))
bspins120 = np.nanmean(absSpindles120, axis=0) + 2.576*np.nanstd(absSpindles120, axis=0)/np.sqrt(len(absSpindles120))

#Plot
fig3b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(absSpindles120,axis=0), color=col)
plt.fill_between(xes, aspins120,bspins120, color=col, alpha=0.4)
plt.axvline(x=lowthre120, color='k', lw=1, ls='--')
plt.axvline(x=highthre120, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## 120s <= REMpre < 180s

#99% Confidence Interval
aspins180 = np.nanmean(absSpindles180, axis=0) - 2.576*np.nanstd(absSpindles180, axis=0)/np.sqrt(len(absSpindles180))
bspins180 = np.nanmean(absSpindles180, axis=0) + 2.576*np.nanstd(absSpindles180, axis=0)/np.sqrt(len(absSpindles180))

#Plot
fig3c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(absSpindles180,axis=0), color=col)
plt.fill_between(xes, aspins180,bspins180, color=col, alpha=0.4)
plt.axvline(x=lowthre180, color='k', lw=1, ls='--')
plt.axvline(x=highthre180, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## 180s <= REMpre < 240s

#99% Confidence Interval
aspins240 = np.nanmean(absSpindles240, axis=0) - 2.576*np.nanstd(absSpindles240, axis=0)/np.sqrt(len(absSpindles240))
bspins240 = np.nanmean(absSpindles240, axis=0) + 2.576*np.nanstd(absSpindles240, axis=0)/np.sqrt(len(absSpindles240))

#Plot
fig3d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(absSpindles240,axis=0), color=col)
plt.fill_between(xes, aspins240,bspins240, color=col, alpha=0.4)
plt.axvline(x=lowthre240, color='k', lw=1, ls='--')
plt.axvline(x=highthre240, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

#Save figures
fig3a.savefig(outpath+'sf3_absSpin60.pdf')
fig3b.savefig(outpath+'sf3_absSpin120.pdf')
fig3c.savefig(outpath+'sf3_absSpin180.pdf')
fig3d.savefig(outpath+'sf3_absSpin240.pdf')


### MAs

## 7.5s <= REMpre < 60s

#99% Confidence Interval
ama60 = np.nanmean(absMAs60, axis=0) - 2.576*np.nanstd(absMAs60, axis=0)/np.sqrt(len(absMAs60))
bma60 = np.nanmean(absMAs60, axis=0) + 2.576*np.nanstd(absMAs60, axis=0)/np.sqrt(len(absMAs60))

#Plot
fig4a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(absMAs60,axis=0), color=col)
plt.fill_between(xes, ama60,bma60, color=col, alpha=0.4)
plt.axvline(x=lowthre60, color='k', lw=1, ls='--')
plt.axvline(x=highthre60, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## 60s <= REMpre < 120s

#99% Confidence Interval
ama120 = np.nanmean(absMAs120, axis=0) - 2.576*np.nanstd(absMAs120, axis=0)/np.sqrt(len(absMAs120))
bma120 = np.nanmean(absMAs120, axis=0) + 2.576*np.nanstd(absMAs120, axis=0)/np.sqrt(len(absMAs120))

#Plot
fig4b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(absMAs120,axis=0), color=col)
plt.fill_between(xes, ama120,bma120, color=col, alpha=0.4)
plt.axvline(x=lowthre120, color='k', lw=1, ls='--')
plt.axvline(x=highthre120, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## 120s <= REMpre < 180s

#99% Confidence Interval
ama180 = np.nanmean(absMAs180, axis=0) - 2.576*np.nanstd(absMAs180, axis=0)/np.sqrt(len(absMAs180))
bma180 = np.nanmean(absMAs180, axis=0) + 2.576*np.nanstd(absMAs180, axis=0)/np.sqrt(len(absMAs180))

#Plot
fig4c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(absMAs180,axis=0), color=col)
plt.fill_between(xes, ama180,bma180, color=col, alpha=0.4)
plt.axvline(x=lowthre180, color='k', lw=1, ls='--')
plt.axvline(x=highthre180, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## 180s <= REMpre < 240s

#99% Confidence Interval
ama240 = np.nanmean(absMAs240, axis=0) - 2.576*np.nanstd(absMAs240, axis=0)/np.sqrt(len(absMAs240))
bma240 = np.nanmean(absMAs240, axis=0) + 2.576*np.nanstd(absMAs240, axis=0)/np.sqrt(len(absMAs240))

#Plot
fig4d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(absMAs240,axis=0), color=col)
plt.fill_between(xes, ama240,bma240, color=col, alpha=0.4)
plt.axvline(x=lowthre240, color='k', lw=1, ls='--')
plt.axvline(x=highthre240, color='k', lw=1, ls='--')
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

#Save figures
fig4a.savefig(outpath+'sf3_absma60.pdf')
fig4b.savefig(outpath+'sf3_absma120.pdf')
fig4c.savefig(outpath+'sf3_absma180.pdf')
fig4d.savefig(outpath+'sf3_absma240.pdf')

###############################################################################
### E - Progression of Delta,Theta,Sigma,Spindle,MAs after wake

#Function that finds transitions from nrem to wake
def nwtransitions(tupList2):
    nwt_locs = []
    cnt1 = 0
    while cnt1 < len(tupList2)-1:
        curr = tupList2[cnt1]
        nxt = tupList2[cnt1+1]
        if (curr[0]=='N')&(nxt[0]=='W'):
            nwt_locs.append(cnt1+1)
        cnt1+=1
    return nwt_locs

#Lists to stores results split by different durations of Wake blocks
w_absDeltas60 = []
w_absDeltas120 = []
w_absDeltas180 = []
w_absDeltas240 = []
w_absDeltasInf = []

w_absThetas60 = []
w_absThetas120 = []
w_absThetas180 = []
w_absThetas240 = []
w_absThetasInf = []

w_absSigmas60 = []
w_absSigmas120 = []
w_absSigmas180 = []
w_absSigmas240 = []
w_absSigmasInf = []

w_absSpindles60 = []
w_absSpindles120 = []
w_absSpindles180 = []
w_absSpindles240 = []
w_absSpindlesInf = []

w_absMAs60 = []
w_absMAs120 = []
w_absMAs180 = []
w_absMAs240 = []
w_absMAsInf = []


twin=3

for rec in spindleRecs:
    #Find locations of spindles
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
    #Load recordings and find cycles
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
        nwt_locs = nwtransitions(tupList2)
        #Loop through all cycles
        cnt2 = 0
        while cnt2 < len(nwt_locs)-1:
            sub = tupList2[nwt_locs[cnt2]:nwt_locs[cnt2+1]]
            wake = sub[0][1]*2.5
            #Find |N| locations (split by 30 second sequences)
            if wake>=7.5:
                states = np.array([x[0] for x in sub])
                remInds = np.where(states=='R')[0]
                if len(remInds)>0:
                    newsub = sub[0:min(remInds)]
                else:
                    newsub = sub
                totlist = list(np.arange(newsub[1][2]-newsub[1][1]+1,newsub[-1][2]+1))
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
                #Calculate all quantities
                allSigmas = []
                allThetas = []
                allDeltas = []
                allMAcnts = []
                allSpincnts = []
                allSWSamt = []
                for subranges in allRanges:
                    if len(subranges)>0:
                        subsigmas = []
                        subdeltas = []
                        subthetas = []
                        subspins = []
                        subMAs = []
                        submacnt = []
                        subspincnt = []
                        subswsamt = []
                        for s in subranges:
                            #Delta,Theta,Sigma powers
                            b = int((s[-1]+1)*nbin)
                            sup = list(range(int(s[0]*nbin),b))
                            if len(sup)>=nwin:
                                Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                df = F[1]-F[0]
                                deltafreq = np.where((F>=0.5)&(F<=4.5))
                                thetafreq = np.where((F>=5)&(F<=9.5))
                                sigmafreq = np.where((F>=10)&(F<=15))
                                deltapow = np.sum(Pow[deltafreq])*df
                                thetapow = np.sum(Pow[thetafreq])*df
                                sigmapow = np.sum(Pow[sigmafreq])*df
                                subdeltas.append(deltapow)
                                subthetas.append(thetapow)
                                subsigmas.append(sigmapow)
                            else:
                                subdeltas.append(np.nan)
                                subthetas.append(np.nan)
                                subsigmas.append(np.nan)
                            #spindles & MA
                            if len(s)>0:
                                #MA
                                seqstart = s[0]
                                seqend = s[-1]
                                subfixvec = fixvec[seqstart:seqend+1]
                                subtup = rp.vecToTup(subfixvec)
                                swsamt = 0
                                macnt = 0
                                for y in subtup:
                                    if y[0]=='MA':
                                        macnt+=1
                                        swsamt+=y[1]*2.5
                                    if y[0]=='N':
                                        swsamt+=y[1]*2.5
                                submacnt.append(macnt)
                                subswsamt.append(swsamt)
                                #Spindle
                                spcnt = 0
                                for y in scaledSpindleTimes:
                                    spinstart = y[0]
                                    spinend = y[1]
                                    if (spinstart>=seqstart)&(spinend<=seqend):
                                        spcnt+=1
                                subspincnt.append(spcnt)
                            else:
                                submacnt.append(np.nan)
                                subspincnt.append(np.nan)
                        allDeltas.append(np.nanmean(subdeltas))
                        allThetas.append(np.nanmean(subthetas))
                        allSigmas.append(np.nanmean(subsigmas))
                        allSpincnts.append(np.nansum(subspincnt))
                        allMAcnts.append(np.nansum(submacnt))
                        allSWSamt.append(sum(subswsamt))
                    else:
                        allDeltas.append(np.nan)
                        allThetas.append(np.nan)
                        allSigmas.append(np.nan)
                        allSpincnts.append(np.nan)
                        allMAcnts.append(np.nan)
                        allSWSamt.append(np.nan)
                #Account for overlaps in spindles and MAs
                for ind,subranges in enumerate(allRanges):
                    if ind<len(allRanges)-1:
                        nxtsubranges = allRanges[ind+1]
                        if (len(subranges)>0)&(len(nxtsubranges)>0):
                            curlast = subranges[-1]
                            nxtfirst = nxtsubranges[0]
                            if nxtfirst[0]==curlast[-1]+1:
                                seqstart = curlast[0]
                                seqend = curlast[-1]
                                seq2start = nxtfirst[0]
                                seq2end = nxtfirst[-1]
                                subfixvec1 = fixvec[seqstart:seqend+1]
                                subfixvec2 = fixvec[seq2start:seq2end+1]
                                subtup1 = rp.vecToTup(subfixvec1)
                                subtup2 = rp.vecToTup(subfixvec2)
                                if (subtup1[-1][0]=='MA')&(subtup2[0][0]=='MA'):
                                    if subtup1[-1][1]>=subtup2[0][1]:
                                        allMAcnts[ind+1] = allMAcnts[ind+1]-1
                                    else:
                                        allMAcnts[ind] = allMAcnts[ind]-1
                                for y in scaledSpindleTimes:
                                    spinstart = y[0]
                                    spinend = y[1]
                                    if (spinstart<=seqend)&(spinend>=seq2start):
                                        if (seqend-spinstart)>=(spinend-seq2start):
                                            allSpincnts[ind] = allSpincnts[ind]+1
                                        else:
                                            allSpincnts[ind+1] = allSpincnts[ind+1]+1
                #Change counts to frequencies (cnt/min.)
                allSpins = []
                allMAs = []
                for ind in range(len(allRanges)):
                    maamt = allMAcnts[ind]
                    spinamt = allSpincnts[ind]
                    if np.isnan(maamt):
                        allSpins.append(np.nan)
                        allMAs.append(np.nan)
                    else:
                        allSpins.append(spinamt/allSWSamt[ind]*60)
                        allMAs.append(maamt/allSWSamt[ind]*60)
                #Add results to lists
                if wake<60:
                    w_absDeltas60.append(allDeltas)
                    w_absThetas60.append(allThetas)
                    w_absSigmas60.append(allSigmas)
                    w_absSpindles60.append(allSpins)
                    w_absMAs60.append(allMAs)
                elif wake<120:
                    w_absDeltas120.append(allDeltas)
                    w_absThetas120.append(allThetas)
                    w_absSigmas120.append(allSigmas)
                    w_absSpindles120.append(allSpins)
                    w_absMAs120.append(allMAs)
                elif wake<180:
                    w_absDeltas180.append(allDeltas)
                    w_absThetas180.append(allThetas)
                    w_absSigmas180.append(allSigmas)
                    w_absSpindles180.append(allSpins)
                    w_absMAs180.append(allMAs)
                elif wake<240:
                    w_absDeltas240.append(allDeltas)
                    w_absThetas240.append(allThetas)
                    w_absSigmas240.append(allSigmas)
                    w_absSpindles240.append(allSpins)
                    w_absMAs240.append(allMAs)
                else:
                    w_absDeltasInf.append(allDeltas)
                    w_absThetasInf.append(allThetas)
                    w_absSigmasInf.append(allSigmas)
                    w_absSpindlesInf.append(allSpins)
                    w_absMAsInf.append(allMAs)                    
            cnt2+=1
    print(rec)

#xvalues to plot
xes = np.arange(15,1015,30)


### Sigma

## 7.5s (20s) <= Wake Block < 60s

#99% Confidence Interval
waSig60 = np.nanmean(w_absSigmas60, axis=0) - 2.576*np.nanstd(w_absSigmas60, axis=0)/np.sqrt(len(w_absSigmas60))
wbSig60 = np.nanmean(w_absSigmas60, axis=0) + 2.576*np.nanstd(w_absSigmas60, axis=0)/np.sqrt(len(w_absSigmas60))

#Plot
fig6a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(w_absSigmas60,axis=0), color=col)
plt.fill_between(xes, waSig60, wbSig60, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

## 60s <= Wake Block < 120s

#99% Confidence Interval
waSig120 = np.nanmean(w_absSigmas120, axis=0) - 2.576*np.nanstd(w_absSigmas120, axis=0)/np.sqrt(len(w_absSigmas120))
wbSig120 = np.nanmean(w_absSigmas120, axis=0) + 2.576*np.nanstd(w_absSigmas120, axis=0)/np.sqrt(len(w_absSigmas120))

#Plot
fig6b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(w_absSigmas120,axis=0), color=col)
plt.fill_between(xes, waSig120, wbSig120, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

## 120s <= Wake Block < 180s

#99% Confidence Interval
waSig180 = np.nanmean(w_absSigmas180, axis=0) - 2.576*np.nanstd(w_absSigmas180, axis=0)/np.sqrt(len(w_absSigmas180))
wbSig180 = np.nanmean(w_absSigmas180, axis=0) + 2.576*np.nanstd(w_absSigmas180, axis=0)/np.sqrt(len(w_absSigmas180))

#Plot
fig6c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(w_absSigmas180,axis=0), color=col)
plt.fill_between(xes, waSig180, wbSig180, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

## 180s <= Wake Block < 240s

#99% Confidence Interval
waSig240 = np.nanmean(w_absSigmas240, axis=0) - 2.576*np.nanstd(w_absSigmas240, axis=0)/np.sqrt(len(w_absSigmas240))
wbSig240 = np.nanmean(w_absSigmas240, axis=0) + 2.576*np.nanstd(w_absSigmas240, axis=0)/np.sqrt(len(w_absSigmas240))

#Plot
fig6d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(w_absSigmas240,axis=0), color=col)
plt.fill_between(xes, waSig240, wbSig240, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

## Wake Block >= 240s

#99% Confidence Interval
waSigInf = np.nanmean(w_absSigmasInf, axis=0) - 2.576*np.nanstd(w_absSigmasInf, axis=0)/np.sqrt(len(w_absSigmasInf))
wbSigInf = np.nanmean(w_absSigmasInf, axis=0) + 2.576*np.nanstd(w_absSigmasInf, axis=0)/np.sqrt(len(w_absSigmasInf))

#Plot
fig6e = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = 'purple'
plt.plot(xes, np.nanmean(w_absSigmasInf,axis=0), color=col)
plt.fill_between(xes, waSigInf, wbSigInf, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([400,1300])
plt.yticks([400,800,1200])

#Save Figures
fig6a.savefig(outpath+'sf3_wakesig60.pdf')
fig6b.savefig(outpath+'sf3_wakesig150.pdf')
fig6c.savefig(outpath+'sf3_wakesig180.pdf')
fig6d.savefig(outpath+'sf3_wakesig240.pdf')
fig6e.savefig(outpath+'sf3_wakesiginf.pdf')


### Thetas

## 7.5s (20s) <= Wake Block < 60s

#99% Confidence Interval
waThe60 = np.nanmean(w_absThetas60, axis=0) - 2.576*np.nanstd(w_absThetas60, axis=0)/np.sqrt(len(w_absThetas60))
wbThe60 = np.nanmean(w_absThetas60, axis=0) + 2.576*np.nanstd(w_absThetas60, axis=0)/np.sqrt(len(w_absThetas60))

#Plot
fig7a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(w_absThetas60,axis=0), color=col)
plt.fill_between(xes, waThe60, wbThe60, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])
plt.yticks([1200,1600,2000,2400])

## 60s <= Wake Block < 120s

#99% Confidence Interval
wathe120 = np.nanmean(w_absThetas120, axis=0) - 2.576*np.nanstd(w_absThetas120, axis=0)/np.sqrt(len(w_absThetas120))
wbthe120 = np.nanmean(w_absThetas120, axis=0) + 2.576*np.nanstd(w_absThetas120, axis=0)/np.sqrt(len(w_absThetas120))

#Plot
fig7b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(w_absThetas120,axis=0), color=col)
plt.fill_between(xes, wathe120,wbthe120, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])
plt.yticks([1200,1600,2000,2400])

## 120s <= Wake Block <180s

#99% Confidence Interval
wathe180 = np.nanmean(w_absThetas180, axis=0) - 2.576*np.nanstd(w_absThetas180, axis=0)/np.sqrt(len(w_absThetas180))
wbthe180 = np.nanmean(w_absThetas180, axis=0) + 2.576*np.nanstd(w_absThetas180, axis=0)/np.sqrt(len(w_absThetas180))

#Plot
fig7c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(w_absThetas180,axis=0), color=col)
plt.fill_between(xes, wathe180,wbthe180, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])
plt.yticks([1200,1600,2000,2400])

## 180s <= Wake Block < 240s

#99% Confidence Interval
wathe240 = np.nanmean(w_absThetas240, axis=0) - 2.576*np.nanstd(w_absThetas240, axis=0)/np.sqrt(len(w_absThetas240))
wbthe240 = np.nanmean(w_absThetas240, axis=0) + 2.576*np.nanstd(w_absThetas240, axis=0)/np.sqrt(len(w_absThetas240))

#Plot
fig7d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(w_absThetas240,axis=0), color=col)
plt.fill_between(xes, wathe240,wbthe240, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])
plt.yticks([1200,1600,2000,2400])

## Wake Block >= 240s

#99% Confidence Interval
watheInf = np.nanmean(w_absThetasInf, axis=0) - 2.576*np.nanstd(w_absThetasInf, axis=0)/np.sqrt(len(w_absThetasInf))
wbtheInf = np.nanmean(w_absThetasInf, axis=0) + 2.576*np.nanstd(w_absThetasInf, axis=0)/np.sqrt(len(w_absThetasInf))

#Plot
fig7e = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = 'purple'
plt.plot(xes, np.nanmean(w_absThetasInf,axis=0), color=col)
plt.fill_between(xes, watheInf,wbtheInf, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([1000,2400])
plt.yticks([1200,1600,2000,2400])

#Save Figures
fig7a.savefig(outpath+'sf3_wakethetas60.pdf')
fig7b.savefig(outpath+'sf3_wakethetas120.pdf')
fig7c.savefig(outpath+'sf3_wakethetas180.pdf')
fig7d.savefig(outpath+'sf3_wakethetas240.pdf')
fig7e.savefig(outpath+'sf3_wakethetasinf.pdf')


### Spindles

## 7.5s (20s) <= Wake Block < 60s

#99% Confidence Interval
waspins60 = np.nanmean(w_absSpindles60, axis=0) - 2.576*np.nanstd(w_absSpindles60, axis=0)/np.sqrt(len(w_absSpindles60))
wbspins60 = np.nanmean(w_absSpindles60, axis=0) + 2.576*np.nanstd(w_absSpindles60, axis=0)/np.sqrt(len(w_absSpindles60))

#Plot
fig8a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(w_absSpindles60,axis=0), color=col)
plt.fill_between(xes, waspins60,wbspins60, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## 60s <= Wake Block < 120s

#99% Confidence Interval
waspins120 = np.nanmean(w_absSpindles120, axis=0) - 2.576*np.nanstd(w_absSpindles120, axis=0)/np.sqrt(len(w_absSpindles120))
wbspins120 = np.nanmean(w_absSpindles120, axis=0) + 2.576*np.nanstd(w_absSpindles120, axis=0)/np.sqrt(len(w_absSpindles120))

#Plot
fig8b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(w_absSpindles120,axis=0), color=col)
plt.fill_between(xes, waspins120,wbspins120, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## 120s <= Wake Block < 180s

#99% Confidence Interval
waspins180 = np.nanmean(w_absSpindles180, axis=0) - 2.576*np.nanstd(w_absSpindles180, axis=0)/np.sqrt(len(w_absSpindles180))
wbspins180 = np.nanmean(w_absSpindles180, axis=0) + 2.576*np.nanstd(w_absSpindles180, axis=0)/np.sqrt(len(w_absSpindles180))

#Plot
fig8c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(w_absSpindles180,axis=0), color=col)
plt.fill_between(xes, waspins180,wbspins180, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## 180s <= Wake Block < 240s

#99% Confidence Interval
waspins240 = np.nanmean(w_absSpindles240, axis=0) - 2.576*np.nanstd(w_absSpindles240, axis=0)/np.sqrt(len(w_absSpindles240))
wbspins240 = np.nanmean(w_absSpindles240, axis=0) + 2.576*np.nanstd(w_absSpindles240, axis=0)/np.sqrt(len(w_absSpindles240))

#Plot
fig8d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(w_absSpindles240,axis=0), color=col)
plt.fill_between(xes, waspins240,wbspins240, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

## Wake Block >= 240s

#99% Confidence Interval
waspinsInf = np.nanmean(w_absSpindlesInf, axis=0) - 2.576*np.nanstd(w_absSpindlesInf, axis=0)/np.sqrt(len(w_absSpindlesInf))
wbspinsInf = np.nanmean(w_absSpindlesInf, axis=0) + 2.576*np.nanstd(w_absSpindlesInf, axis=0)/np.sqrt(len(w_absSpindlesInf))

#Plot
fig8e = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = 'purple'
plt.plot(xes, np.nanmean(w_absSpindlesInf,axis=0), color=col)
plt.fill_between(xes, waspinsInf,wbspinsInf, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,6.5])
plt.yticks([0,2,4,6])

#Save Figures
fig8a.savefig(outpath+'sf3_wakespins60.pdf')
fig8b.savefig(outpath+'sf3_wakespins120.pdf')
fig8c.savefig(outpath+'sf3_wakespins180.pdf')
fig8d.savefig(outpath+'sf3_wakespins240.pdf')
fig8e.savefig(outpath+'sf3_wakespinsinf.pdf')


### MAs

## 7.5s (20s) <= Wake Block < 60s

#99% Confidence Interval
wama60 = np.nanmean(w_absMAs60, axis=0) - 2.576*np.nanstd(w_absMAs60, axis=0)/np.sqrt(len(w_absMAs60))
wbma60 = np.nanmean(w_absMAs60, axis=0) + 2.576*np.nanstd(w_absMAs60, axis=0)/np.sqrt(len(w_absMAs60))

#Plot
fig9a = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(1)
plt.plot(xes, np.nanmean(w_absMAs60,axis=0), color=col)
plt.fill_between(xes, wama60,wbma60, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## 60s <= Wake Block < 120s

#99% Confidence Interval
wama120 = np.nanmean(w_absMAs120, axis=0) - 2.576*np.nanstd(w_absMAs120, axis=0)/np.sqrt(len(w_absMAs120))
wbma120 = np.nanmean(w_absMAs120, axis=0) + 2.576*np.nanstd(w_absMAs120, axis=0)/np.sqrt(len(w_absMAs120))

#Plot
fig9b = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(3)
plt.plot(xes, np.nanmean(w_absMAs120,axis=0), color=col)
plt.fill_between(xes, wama120,wbma120, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## 120s <= Wake Block < 180s

#99% Confidence Interval
wama180 = np.nanmean(w_absMAs180, axis=0) - 2.576*np.nanstd(w_absMAs180, axis=0)/np.sqrt(len(w_absMAs180))
wbma180 = np.nanmean(w_absMAs180, axis=0) + 2.576*np.nanstd(w_absMAs180, axis=0)/np.sqrt(len(w_absMAs180))

#Plot
fig9c = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(5)
plt.plot(xes, np.nanmean(w_absMAs180,axis=0), color=col)
plt.fill_between(xes, wama180,wbma180, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## 180s <= Wake Block < 240s

#99% Confidence Interval
wama240 = np.nanmean(w_absMAs240, axis=0) - 2.576*np.nanstd(w_absMAs240, axis=0)/np.sqrt(len(w_absMAs240))
wbma240 = np.nanmean(w_absMAs240, axis=0) + 2.576*np.nanstd(w_absMAs240, axis=0)/np.sqrt(len(w_absMAs240))

#Plot
fig9d = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = scalarMap.to_rgba(7)
plt.plot(xes, np.nanmean(w_absMAs240,axis=0), color=col)
plt.fill_between(xes, wama240,wbma240, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

## Wake Block >= 240s 

#99% Confidence Interval
wamaInf = np.nanmean(w_absMAsInf, axis=0) - 2.576*np.nanstd(w_absMAsInf, axis=0)/np.sqrt(len(w_absMAsInf))
wbmaInf = np.nanmean(w_absMAsInf, axis=0) + 2.576*np.nanstd(w_absMAsInf, axis=0)/np.sqrt(len(w_absMAsInf))

#Plot
fig9e = plt.figure()
cmap = plt.get_cmap('YlOrRd')
cNorm = colors.Normalize(vmin=0, vmax=10)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
col = 'purple'
plt.plot(xes, np.nanmean(w_absMAsInf,axis=0), color=col)
plt.fill_between(xes, wamaInf,wbmaInf, color=col, alpha=0.4)
sns.despine()
plt.xlim([0,600])
plt.xticks([0,250,500])
plt.ylim([0,2.5])
plt.yticks([0,1,2])

#Save Figures
fig9a.savefig(outpath+'sf3_wakemas60.pdf')
fig9b.savefig(outpath+'sf3_wakemas120.pdf')
fig9c.savefig(outpath+'sf3_wakemas180.pdf')
fig9d.savefig(outpath+'sf3_wakemas240.pdf')
fig9e.savefig(outpath+'sf3_wakemasinf.pdf')

