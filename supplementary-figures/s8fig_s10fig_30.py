#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:37:42 2021

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
import scipy.io as so


#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/nrev_figs/ma30/'


#Define colors
current_palette = sns.color_palette('muted', 10)
col1 = current_palette[3]
col2 = current_palette[0]

#Make dataframe containing all REM-NREM cycles
remDF = rp.standard_recToDF(ppath, recordings, 12, False) #MA threshold = 30 s
remDF = remDF.loc[remDF.rem<240]
nonDF = remDF.loc[remDF.rem<7.5]
subDF = remDF.loc[remDF.rem>=7.5]
subDF = subDF.reset_index(drop=True)


#Read in GMM parameters and model coefficients
gmmDF = pd.read_csv('gmm30DF.csv')


khigh = gmmDF.khigh
mlow = gmmDF.mlow
mhigh = gmmDF.mhigh
slow = gmmDF.slow
shigh = gmmDF.shigh


from scipy.optimize import curve_fit

#Estimate log and linear fit parameters
#Define log function to model change of parameters
def log_func(x,a,b,c):
    return a*np.log(x+b)+c

xes = np.arange(15,240,30)

#Fit coefficients of log function to each parameter
khigh_log = curve_fit(log_func, xes[0:6], khigh[0:6], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]))[0]
khigh_loga = khigh_log[0]
khigh_logb = khigh_log[1]
khigh_logc = khigh_log[2]

mlow_log = curve_fit(log_func, xes[0:5], mlow[0:5], p0=[-1,1,1], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]), maxfev=9999999)[0]
mlow_loga = mlow_log[0]
mlow_logb = mlow_log[1]
mlow_logc = mlow_log[2]

mhigh_log = curve_fit(log_func, xes, mhigh,bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]))[0]
mhigh_loga = mhigh_log[0]
mhigh_logb = mhigh_log[1]
mhigh_logc = mhigh_log[2]

slow_log = curve_fit(log_func, xes[0:5], slow[0:5], p0=[-1,1,1], bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
slow_loga = slow_log[0]
slow_logb = slow_log[1]
slow_logc = slow_log[2]

shigh_log = curve_fit(log_func, xes, shigh, p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
shigh_loga = shigh_log[0]
shigh_logb = shigh_log[1]
shigh_logc = shigh_log[2]

#Fit coefficients of linear function to each parameter
xplot1 = np.arange(0,240)
xplot2 = np.arange(0,160)

import statsmodels.api as sm


#slow
x3 = np.arange(15,240,30)[0:5]
x3 = sm.add_constant(x3)
y3 = slow[0:5]
mod3 = sm.OLS(y3,x3)
res3 = mod3.fit()
linreg3 = lambda x: res3.params[1]*x + res3.params[0]



###############################################################################

### Boundaries

slow_lina = res3.params[1]
slow_linb = res3.params[0]

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

from scipy.optimize import fsolve
from scipy import stats
import scipy.io as so

###Figure 3F
#Spectral Density comparing REM and |N| of Sequential and Single cycles


##Parietal EEG
seqSwsSpec = []
sinSwsSpec = []

twin=3

#Go through every REM-NREM cycle (in every recording) with REM in [7.5,240]
#and calculate spectral density for REM and |N|.
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
            fixtup = rp.ma_thr(Mtup, 12)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 12)
        Mtups.append(fixtup)
    for tupList in Mtups:
        tupList2 = tupList[1:len(tupList)-1]
        nrt_locs = rp.nrt_locations(tupList2)
        cnt2 = 0
        #Loop through all cycles
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            #Find location of REM and |N| of cycles
            states = np.array([x[0] for x in sub])
            swsInds = np.where((states=='N')|(states=='MA'))[0]
            swsSeqs = rp.stateSeq(sub, swsInds)
            #Calculate spectral density
            subSwsSpec = []
            for s in swsSeqs:
                if len(s)*nbin>=nwin:
                    b = int((s[-1]+1)*nbin)
                    sup = list(range(int(s[0]*nbin), b))
                    if sup[-1]>len(EEG):
                        sup = list(range(int(s[0]*nbin), len(EEG)))
                    if len(sup) >= nwin:
                        Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                        subSwsSpec.append(Pow)
            ifreq = np.where(F<=20)
            swsSpec = np.array([x[ifreq] for x in subSwsSpec])
            avgSwsSpec = np.mean(swsSpec,axis=0)
            rem = sub[0][1]*2.5
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240]
                if rp.isSequential(rem,sws,intersect_x,intersect_y): #Check if sequential or single
                    seqSwsSpec.append(avgSwsSpec)
                else:
                    sinSwsSpec.append(avgSwsSpec)
            cnt2+=1
    print(rec)

#Limit inspection to frequencies in btw [0Hz,20Hz]                
ifreq = np.where(F<=20)
FN = F[ifreq]

seqSwsS = np.array(seqSwsSpec)
sinSwsS = np.array(sinSwsSpec)

#Calculate 99% confidence intervals
aseqSws = seqSwsS.mean(axis=0) - 2.576*seqSwsS.std(axis=0)/np.sqrt(len(seqSwsS))
bseqSws = seqSwsS.mean(axis=0) + 2.576*seqSwsS.std(axis=0)/np.sqrt(len(seqSwsS))
asinSws = sinSwsS.mean(axis=0) - 2.576*sinSwsS.std(axis=0)/np.sqrt(len(sinSwsS))
bsinSws = sinSwsS.mean(axis=0) + 2.576*sinSwsS.std(axis=0)/np.sqrt(len(sinSwsS))

##stats
eeg1SWSFreq = []
eeg1SWSwpval = [] #welch's test

#Perform welch's test for eaach frequency between [0,15]Hz
for i in range(46):
    curfreqind = i
    seqp = np.array([x[curfreqind] for x in seqSwsS])
    sinp = np.array([x[curfreqind] for x in sinSwsS])
    eeg1SWSFreq.append(F[i])
    eeg1SWSwpval.append(stats.ttest_ind(seqp,sinp,equal_var=False)[1])

#Store frequencies for which difference is significant at alpha=0.05/0.01/0.001
eeg1sws_sig1 = []
eeg1sws_sig2 = []
eeg1sws_sig3 = []

for idx,x in enumerate(eeg1SWSwpval):
    if x<0.05:
        eeg1sws_sig1.append(eeg1SWSFreq[idx])
    if x<0.01:
        eeg1sws_sig2.append(eeg1SWSFreq[idx])
    if x<0.001:
        eeg1sws_sig3.append(eeg1SWSFreq[idx])

#significant locs
eeg1sws_sig3f1 = np.arange(0.6666, 5.3333, 0.0001)
eeg1sws_sig3f2 = np.arange(6, 7.3333, 0.0001)
eeg1sws_sig3f3 = np.arange(8.3333, 15, 0.0001)


fig3F2 = plt.figure()
plt.plot(FN, seqSwsS.mean(axis=0), label='Sequential', color=col2, lw=1)
plt.plot(FN, sinSwsS.mean(axis=0), label='Single', color=col1, lw=1)
plt.fill_between(FN, asinSws,bsinSws, color=col1, alpha=0.5)
plt.fill_between(FN, aseqSws,bseqSws, color=col2, alpha=0.5)
plt.plot(eeg1sws_sig3f1, np.repeat(1100,len(eeg1sws_sig3f1)),lw=1,color='k')
plt.plot(eeg1sws_sig3f2, np.repeat(1100,len(eeg1sws_sig3f2)),lw=1,color='k')
plt.plot(eeg1sws_sig3f3, np.repeat(1100,len(eeg1sws_sig3f3)),lw=1,color='k')
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,1300])

fig3F2.savefig(outpath+'fig3F2.pdf')
plt.close(fig3F2)



#####Prefrontal EEG

tseqSwsSpec2 = []
tsinSwsSpec2 = []

twin=3

#Exclude recording that has no Prefrontal EEG
recordings2 = [x for x in recordings if x!='HA41_081618n1']

for rec in recordings2:
    #Load recording and define cycles
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
            fixtup = rp.ma_thr(Mtup, 12)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 12)
        Mtups.append(fixtup)
    for tupList in Mtups:
        tupList2 = tupList[1:len(tupList)-1]
        nrt_locs = rp.nrt_locations(tupList2)
        cnt2 = 0
        #Loop through cycles
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            #Find location of REM and |N| within cycles
            states = np.array([x[0] for x in sub])
            swsInds = np.where((states=='N')|(states=='MA'))[0]
            swsSeqs = rp.stateSeq(sub, swsInds)
            #Calculate Spectral density
            swsSpec = []
            subSwsSpec = []
            for s in swsSeqs:
                if len(s)*nbin>=nwin:
                    b = int((s[-1]+1)*nbin)
                    sup = list(range(int(s[0]*nbin), b))
                    if sup[-1]>len(EEG):
                        sup = list(range(int(s[0]*nbin), len(EEG)))
                    if len(sup) >= nwin:
                        Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                        swsSpec.append(Pow)
                        subSwsSpec.append(Pow)
            ifreq = np.where(F<=20)
            tswsSpec = np.array([x[ifreq] for x in subSwsSpec])
            avgswsSpec = np.mean(tswsSpec,axis=0)
            rem = sub[0][1]*2.5
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240]
                if rp.isSequential(rem,sws,intersect_x,intersect_y): #Check if cycle is sequential or single
                    tseqSwsSpec2.append(avgswsSpec)
                else:
                    tsinSwsSpec2.append(avgswsSpec)
            cnt2+=1
    print(rec)

#Limit inspection to frequencies in btw [0Hz,20Hz]                   
ifreq = np.where(F<=20)
FN = F[ifreq]

seqSwsS2 = np.array(tseqSwsSpec2)
sinSwsS2 = np.array(tsinSwsSpec2)

#Calculate 99% confidence intervals
aseqSws2 = seqSwsS2.mean(axis=0) - 2.576*seqSwsS2.std(axis=0)/np.sqrt(len(seqSwsS2))
bseqSws2 = seqSwsS2.mean(axis=0) + 2.576*seqSwsS2.std(axis=0)/np.sqrt(len(seqSwsS2))
asinSws2 = sinSwsS2.mean(axis=0) - 2.576*sinSwsS2.std(axis=0)/np.sqrt(len(sinSwsS2))
bsinSws2 = sinSwsS2.mean(axis=0) + 2.576*sinSwsS2.std(axis=0)/np.sqrt(len(sinSwsS2))


##Stats
eeg2SWSFreq = []
eeg2SWSwpval = [] #welch's test

for i in range(46):
    curfreqind = i
    seqp = np.array([x[curfreqind] for x in tseqSwsSpec2])
    sinp = np.array([x[curfreqind] for x in tsinSwsSpec2])
    eeg2SWSFreq.append(F[i])
    eeg2SWSwpval.append(stats.ttest_ind(seqp,sinp,equal_var=False)[1])

eeg2sws_sig1 = []
eeg2sws_sig2 = []
eeg2sws_sig3 = []

for idx,x in enumerate(eeg2SWSwpval):
    if x<0.05:
        eeg2sws_sig1.append(eeg2SWSFreq[idx])
    if x<0.01:
        eeg2sws_sig2.append(eeg2SWSFreq[idx])
    if x<0.001:
        eeg2sws_sig3.append(eeg2SWSFreq[idx])
        
eeg2sws_sig3f1 = np.arange(0.6666, 15, 0.0001)


fig3F4 = plt.figure()
plt.plot(FN, seqSwsS2.mean(axis=0), label='Sequential', color=col2, lw=1)
plt.plot(FN, sinSwsS2.mean(axis=0), label='Single', color=col1, lw=1)
plt.fill_between(FN, aseqSws2, bseqSws2, color=col2, alpha=0.5)
plt.fill_between(FN, asinSws2, bsinSws2, color=col1, alpha=0.5)
plt.plot(eeg2sws_sig3f1, np.repeat(1100,len(eeg2sws_sig3f1)),lw=1,color='k')
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,1225])


fig3F4.savefig(outpath+'fig3F4.pdf')

plt.close(fig3F4)

###############################################################################


#Find recordigns with Spindles
spindleRecs = []
for rec in recordings:
    if os.path.isfile(ppath+rec+'/'+'vip_'+rec+'.txt'):
        spindleRecs.append(rec)


#Lists to store results for different lengths of REMpre
mas60 = []
mas120 = []
mas180 = []
mas240 = []

sigmapows60 = []
sigmapows120 = []
sigmapows180 = []
sigmapows240 = []

thetapows60 = []
thetapows120 = []
thetapows180 = []
thetapows240 = []

twin=3

#Loop through all single cycles in all recordings and calculate the progression
#of sigma,theta,delta,spindles,MAs through quarters of Refractory and quarters
#of permissive period. Do this for different lengths of REMpre.

for rec in spindleRecs:
    #Load recording and find cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    M,S = rp.load_stateidx(ppath, rec)
    revvec = rp.nts(M)
    revtup = rp.vecToTup(revvec)
    revfixtup = rp.ma_thr(revtup, 12)
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
            fixtup = rp.ma_thr(Mtup, 12)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 12)
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

                            subsigmapows = []
                            subthetapows = []
                            #Delta,Sigma,Theta
                            for s in quarter:
                                if len(s)*nbin>=nwin:
                                    b = int((s[-1]+1)*nbin)
                                    sup = list(range(int(s[0]*nbin),b))
                                    if sup[-1]>len(EEG):
                                        sup = list(range(int(s[0]*nbin), len(EEG)))
                                    if len(sup) >= nwin:
                                        Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                        df = F[1]-F[0]
                                        sigmafreq = np.where((F>=10)&(F<=15))
                                        thetafreq = np.where((F>=5)&(F<=9.5))
                                        sigmapow = np.sum(Pow[sigmafreq])*df
                                        thetapow = np.sum(Pow[thetafreq])*df
                                        subsigmapows.append(sigmapow)
                                        subthetapows.append(thetapow)
                                    else:
                                        subsigmapows.append(np.nan)
                                        subthetapows.append(np.nan)
                                else:
                                    subsigmapows.append(np.nan)
                                    subthetapows.append(np.nan)
                            sigmapows.append(np.nanmean(subsigmapows))
                            thetapows.append(np.nanmean(subthetapows))
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
                            marates.append(macnt/nremamt*60)
                        mname = rec.split('_')[0]
                        #Add results to lists
                        if rem<60:
                            thetapows60.append(thetapows)
                            sigmapows60.append(sigmapows)
                            mas60.append(marates)
                        elif rem<120:
                            thetapows120.append(thetapows)
                            sigmapows120.append(sigmapows)
                            mas120.append(marates)       
                        elif rem<180:
                            thetapows180.append(thetapows)
                            sigmapows180.append(sigmapows)
                            mas180.append(marates)
                        else:
                            thetapows240.append(thetapows)
                            sigmapows240.append(sigmapows)
                            mas240.append(marates)
            cnt2+=1
    print(rec)

allThetas = [thetapows60,thetapows120,thetapows180,thetapows240]
allSigmas = [sigmapows60,sigmapows120,sigmapows180,sigmapows240]
allMAs = [mas60,mas120,mas180,mas240]

#Compile results regardless of REMpre
sigscomp = []
for x in allSigmas:
    sigscomp.extend(x)
mascomp = []
for x in allMAs:
    mascomp.extend(x)
thescomp = []
for x in allThetas:
    thescomp.extend(x)

#xvalue to plot
xes = np.arange(0.5,8,1)

##Theta Power

#99% Conf. Interval
aThe = np.nanmean(thescomp,axis=0) - 2.576*np.nanstd(thescomp,axis=0)/np.sqrt(len(thescomp))
bThe = np.nanmean(thescomp,axis=0) + 2.576*np.nanstd(thescomp,axis=0)/np.sqrt(len(thescomp))

#Plot
fig4F1 = plt.figure()
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
plt.ylim([900,2200])

fig4F1.savefig(outpath+'fig4F_eeg2theta2.pdf')

plt.close(fig4F1)


##Sigma Power

#99% Conf. Interval
aSig = np.nanmean(sigscomp,axis=0) - 2.576*np.nanstd(sigscomp,axis=0)/np.sqrt(len(sigscomp))
bSig = np.nanmean(sigscomp,axis=0) + 2.576*np.nanstd(sigscomp,axis=0)/np.sqrt(len(sigscomp))

#Plot
fig4F2 = plt.figure()
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
plt.ylim([350,1210])

fig4F2.savefig(outpath+'fig4F_eeg2sigma2.pdf')

plt.close(fig4F2)

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
plt.ylim([0,3.5])

fig4G2.savefig(outpath+'fig4G_rel_ma2.pdf')

plt.close(fig4G2)

###############################################################################

### Fig 5B - Single Cycles with and without wake

#Divide single cycles to those with wake and without wake
sinwwDF = sinDF.loc[sinDF.wake>0]
sinwoDF = sinDF.loc[sinDF.wake==0]

#Colors and values for boxplot
labels = ['|W|=0','|W|>0']
sizes2 = [len(sinwoDF), len(sinwwDF)]
cols2 = ['mistyrose',col1]


#Boxplot to compare |N|
fig5B2,ax = plt.subplots(figsize=[4,4.8])
bp = ax.boxplot([sinwoDF.sws,sinwwDF.sws],positions=[0.3,0.6],widths=0.15, vert=True, patch_artist=True)
for ind,patch in enumerate(bp['boxes']):
    patch.set_facecolor(cols2[ind])
sns.despine()
plt.xticks([0.3,0.6],labels)
plt.xlim([0.15,0.75])
plt.ylim([0,3100])

#Save figures
fig5B2.savefig(outpath+'f5b_sinboxplot.pdf')
plt.close(fig5B2)

#Statistics
stats.levene(sinwoDF.sws,sinwwDF.sws)
stats.ttest_ind(sinwoDF.sws,sinwwDF.sws,equal_var=False)

############################

### Fig 5C - Distribution of # of wake blocks within single cycles

#List to store # of wake blocks
wakecounts = []

#Loop through recordings and count # of wake blocks in single cycles
for rec in recordings: 
    #Load recordings and find cycles
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    Mtups = []
    for idx,x in enumerate(recs):
        Mvec = rp.nts(x)
        if idx==0:
            Mtup = rp.vecToTup(Mvec, start=0)
            fixtup = rp.ma_thr(Mtup, 12)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 12)
        Mtups.append(fixtup)
    for tupList in Mtups:
        tupList2 = tupList[1:len(tupList)-1]
        nrt_locs = rp.nrt_locations(tupList2)
        #Loop through cycles
        cnt2 = 0
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            for y in sub:
                if (y[0]=='N')or(y[0]=='MA'):
                    sws+=y[1]*2.5
            if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240)
                if rp.isSequential(rem,sws, intersect_x,intersect_y)==False: #Check if cycle is single          
                    wakecnt = 0
                    for x in sub:
                        if x[0]=='W':
                            wakecnt+=1
                    wakecounts.append(wakecnt)
            cnt2+=1
    print(rec)

#Count
wcnt0 = 0
wcnt1 = 0
wcnt2 = 0
wcnt3 = 0
wcnt4 = 0
wcnt5 = 0
wcnt6 = 0
wcnt7 = 0
wcnt8 = 0
wcnt9 = 0
wcnt10 = 0
wcnt11 = 0
wcnt12 = 0
wcnt13 = 0

for x in wakecounts:
    if x==0:
        wcnt0+=1
    elif x==1:
        wcnt1+=1
    elif x==2:
        wcnt2+=1
    elif x==3:
        wcnt3+=1
    elif x==4:
        wcnt4+=1
    elif x==5:
        wcnt5+=1
    elif x==6:
        wcnt6+=1
    elif x==7:
        wcnt7+=1
    elif x==8:
        wcnt8+=1
    elif x==9:
        wcnt9+=1
    elif x==10:
        wcnt10+=1
    elif x==11:
        wcnt11+=1
    elif x==12:
        wcnt12+=1
    else:
        wcnt13+=1

#Put counts in list
allWakeCounts = [wcnt0,wcnt1,wcnt2,wcnt3,wcnt4,wcnt5,wcnt6,wcnt7,wcnt8,wcnt9,wcnt10,wcnt11,wcnt12,wcnt13]

#Change counts to percentage
totalcnt = sum(allWakeCounts)
allWakePercs = [x/totalcnt*100 for x in allWakeCounts]

#Plot
fig5C = plt.figure()
plt.bar([0,1,2,3,4,5,6,7,8,9,10,11,12,13],allWakePercs,color=col1)
sns.despine()
plt.xlabel('# Wake Blocks')
plt.ylabel('% of Single Cycles')

#Save figure
fig5C.savefig(outpath+'wakefreq.pdf')

plt.close(fig5C)


#################################

####Fig 5D - |N| by |W| and #wake blocks

import pingouin

#Lists to store results
wakecounts = []
totwakes = []
rems = []
swss = []
wakeblocks = []

#Loop through recordings, find single cycles and calculate |N|,|W|,# wake blocks
for rec in recordings: 
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    Mtups = []
    for idx,x in enumerate(recs):
        Mvec = rp.nts(x)
        if idx==0:
            Mtup = rp.vecToTup(Mvec, start=0)
            fixtup = rp.ma_thr(Mtup, 12)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 12)
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

#compile results into dataframe
wakeDF = pd.DataFrame(list(zip(rems,swss,wakecounts,totwakes)),columns=['rem','sws','wcount','twake'])
ww_wakeDF = wakeDF.loc[wakeDF.twake>0]
wo_wakeDF = wakeDF.loc[wakeDF.twake==0]

#Find percentiles of |W|
wperc20 = np.percentile(ww_wakeDF.twake,20)
wperc40 = np.percentile(ww_wakeDF.twake,40)
wperc60 = np.percentile(ww_wakeDF.twake,60)
wperc80 = np.percentile(ww_wakeDF.twake,80)

#Add percentile groups to dataframe
wpercGroups = []

for x in wakeDF.twake:
    if x==0:
        wpercGroups.append(0)
    elif x<wperc20:
        wpercGroups.append(1)
    elif x<wperc40:
        wpercGroups.append(2)
    elif x<wperc60:
        wpercGroups.append(3)
    elif x<wperc80:
        wpercGroups.append(4)
    else:
        wpercGroups.append(5)

wakeDF['wpGroup'] = wpercGroups

## |N| vs |W|

#Prepare lists to plot with matplotlib
twake0 = wakeDF.loc[wakeDF.wpGroup==0]
twake1 = wakeDF.loc[wakeDF.wpGroup==1]
twake2 = wakeDF.loc[wakeDF.wpGroup==2]
twake3 = wakeDF.loc[wakeDF.wpGroup==3]
twake4 = wakeDF.loc[wakeDF.wpGroup==4]
twake5 = wakeDF.loc[wakeDF.wpGroup==5]
swsbytwake = [twake0.sws,twake1.sws,twake2.sws,twake3.sws,twake4.sws,twake5.sws]
xpos = [-10, 10, 30, 50, 70, 90]

#Plot
fig5D1 = plt.figure()
bp = plt.boxplot(swsbytwake, positions=xpos, vert=True, patch_artist=True, widths=10)
for patch in bp['boxes']:
    patch.set_facecolor(col1)
sns.despine()
plt.xlabel('|W| (percentile group)')
plt.ylabel('|N| (s)')
plt.ylim([0,3100])

pingouin.homoscedasticity(data=wakeDF,dv="sws",group="wpGroup")
pingouin.welch_anova(data=wakeDF,dv="sws",between="wpGroup")


## |N| vs #Wake blocks

#Prepare lists to plot with matplotlib
wcount0 = wakeDF.loc[wakeDF.wcount==0]
wcount1 = wakeDF.loc[wakeDF.wcount==1]
wcount2 = wakeDF.loc[wakeDF.wcount==2]
wcount3 = wakeDF.loc[wakeDF.wcount==3]
wcount4 = wakeDF.loc[wakeDF.wcount==4]
wcount5 = wakeDF.loc[wakeDF.wcount==5]
wcount6 = wakeDF.loc[wakeDF.wcount==6]
wcount7 = wakeDF.loc[wakeDF.wcount==7]
wcount8 = wakeDF.loc[wakeDF.wcount==8]
wcount9 = wakeDF.loc[wakeDF.wcount==9]
wcount10 = wakeDF.loc[wakeDF.wcount==10]
wcount11 = wakeDF.loc[wakeDF.wcount==11]
wcount12 = wakeDF.loc[wakeDF.wcount==12]
wcount13 = wakeDF.loc[wakeDF.wcount==13]

swsbywcount = [wcount0.sws,wcount1.sws,wcount2.sws,wcount3.sws,wcount4.sws,wcount5.sws,wcount6.sws,wcount7.sws,wcount8.sws,wcount9.sws,wcount10.sws,wcount11.sws,wcount12.sws,wcount13.sws]
xpos2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

#Plot
fig5D2 = plt.figure()
bp = plt.boxplot(swsbywcount, positions=xpos2, vert=True, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor(col1)
sns.despine()
plt.xlabel('# Wake Blocks')
plt.ylabel('|N| (s)')
plt.ylim([0,3100])


#For Wake count up to 5
subwcDF = wakeDF.loc[wakeDF.wcount<6]
swsbywcount2 = [wcount0.sws,wcount1.sws,wcount2.sws,wcount3.sws,wcount4.sws,wcount5.sws]
xpos3 = [0,1,2,3,4,5]

fig5D2b = plt.figure()
bp = plt.boxplot(swsbywcount2, positions=xpos3, vert=True, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor(col1)
sns.despine()
plt.xlabel('# Wake Blocks')
plt.ylabel('|N| (s)')
plt.ylim([0,3100])

pingouin.homoscedasticity(data=subwcDF, dv="sws", group="wcount")
pingouin.welch_anova(data=subwcDF, dv="sws", between="wcount")

#Save figures
fig5D1.savefig(outpath+'f5d_swsbywake.pdf')
fig5D2.savefig(outpath+'f5d_swsbywcount.pdf')
fig5D2b.savefig(outpath+'f5d_swsbywcount2.pdf')

plt.close(fig5D1)
plt.close(fig5D2)
plt.close(fig5D2b)


###############################

### Fig5 E - Progression of Sigma, Theta, Spindles, MAs before & after wake
### on relative time scale

#Function to determine if a sequence contains a wake block
def subHasWake(subsub):
    res = False
    for x in subsub:
        if x[0]=='W':
            res=True
    return res

#Find recordings with spindles
spindleRecs = []
for rec in recordings:
    if os.path.isfile(ppath+rec+'/'+'vip_'+rec+'.txt'):
        spindleRecs.append(rec)

#Lists to store results
allSigs2 = []
allSpins2 = []
allMAs2 = []
allThes2 = []

twin=3

#Loop through all single cycles in all recordings and calculate quantities 
#before and after a wake block
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
    revfixtup = rp.ma_thr(revtup, 12)
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
            fixtup = rp.ma_thr(Mtup, 12)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 12)
        Mtups.append(fixtup)
    for tupList in Mtups:
        tupList2 = tupList[1:len(tupList)-1]
        nrt_locs = rp.nrt_locations(tupList2)
        #Loop through all cycles
        cnt2 = 0
        while cnt2 < len(nrt_locs)-1:
            sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
            rem = sub[0][1]*2.5
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240): #Limit to cycles with rempre in [7.5,240)
                if rp.isSequential(rem,sws,intersect_x,intersect_y)==False:#Check if single cycle
                    states = np.array([x[0] for x in sub])
                    wakelocs = np.where(states=='W')[0]
                    if len(wakelocs)>0:
                        #Find before and after wake
                        befSeqs = []
                        aftSeqs = []
                        wakes = []
                        for x in wakelocs:
                            if x>1: #Exclude REM-NREM cycles that have inter-REM starting with wake
                                if subHasWake(sub[0:x]):
                                    befWakeInd = max(np.where(np.array(sub[0:x])=='W')[0])
                                    befStart = sub[befWakeInd]
                                    befseqstart = befStart[2]+1
                                    befseqend = sub[x][2]-sub[x][1]
                                    
                                    if subHasWake(sub[x+1:]):
                                        aftWakeInd = min(np.where(np.array(sub[x+1:])=='W')[0])+x+1
                                        aftseqstart = sub[x+1][2]-sub[x+1][1]+1
                                        aftseqend = sub[aftWakeInd-1][2]
                                    else:
                                        aftseqstart = sub[x+1][2] - sub[x+1][1]+1
                                        aftseqend = sub[-1][2]
                                else:
                                    befseqstart = sub[0][2]+1
                                    befseqend = sub[x][2]-sub[x][1]
                                    
                                    if subHasWake(sub[x+1:]):
                                        aftWakeInd = min(np.where(np.array(sub[x+1:])=='W')[0])+x+1
                                        aftseqstart = sub[x+1][2]-sub[x+1][1]+1
                                        aftseqend = sub[aftWakeInd-1][2]
                                    else:
                                        aftseqstart = sub[x+1][2]-sub[x+1][1]+1
                                        aftseqend = sub[-1][2]
                            
                                beforeseq = np.arange(befseqstart, befseqend+1)
                                afterseq = np.arange(aftseqstart, aftseqend+1)
                                befSeqs.append(beforeseq)
                                aftSeqs.append(afterseq)
                                wakes.append(sub[x][1]*2.5)
                        # Go through sequences and calculate quantities
                        if len(befSeqs)>0:
                            for ind,befseq in enumerate(befSeqs):
                                aftseq = aftSeqs[ind]
                                curwake = wakes[ind]
                                befses = []
                                aftses = []
                                revbef = befseq[::-1]
                                if (len(befseq)>24)&(len(aftseq)>24):
                                    befsplitlen = int(len(befseq)/4)
                                    aftsplitlen = int(len(aftseq)/4)
                                    for i in range(4):
                                        if i<3:
                                            befses.append(revbef[i*befsplitlen:(i+1)*befsplitlen][::-1])
                                            aftses.append(aftseq[i*aftsplitlen:(i+1)*aftsplitlen])
                                        else:
                                            befses.append(revbef[i*befsplitlen:][::-1])
                                            aftses.append(aftseq[i*aftsplitlen:])
                                    befafts = []
                                    befafts.extend(befses[::-1])
                                    befafts.extend(aftses)
                                    #Sigma and Theta power
                                    Sigmas = []
                                    Thetas = []
                                    for s in befafts:
                                        if len(s)==0:
                                            Sigmas.append(np.nan)
                                            Thetas.append(np.nan)
                                        else:
                                            b = int((s[-1]+1)*nbin)
                                            sup = list(range(int(s[0]*nbin),b))
                                            Pow,F = rp.power_spectrum(EEG[sup],nwin,1/sr)
                                            df = F[1]-F[0]
                                            sigmafreq = np.where((F>=10)&(F<=15))
                                            thetafreq = np.where((F>=5)&(F<=9.5))
                                            sigmapow = np.sum(Pow[sigmafreq])*df
                                            thetapow = np.sum(Pow[thetafreq])*df
                                            Sigmas.append(sigmapow)
                                            Thetas.append(thetapow)
                                    #MAs
                                    macounts = []
                                    marates = []
                                    for s in befafts:
                                        if len(s)==0:
                                            macounts.append(np.nan)
                                        else:
                                            seqstart = s[0]
                                            seqend = s[-1]
                                            subfixvec = fixvec[seqstart:seqend+1]
                                            subtup = rp.vecToTup(subfixvec)
                                            macnt = 0
                                            for y in subtup:
                                                if y[0]=='MA':
                                                    macnt+=1
                                            macounts.append(macnt)
                                    for idx,s in enumerate(befafts):
                                        if idx<len(befafts)-1:
                                            s2 = befafts[idx+1]
                                            if (len(s)>0)&(len(s2)>0):
                                                seqstart = s[0]
                                                seqend = s[-1]
                                                seq2start = s2[0]
                                                seq2end = s2[-1]
                                                subfixvec1 = fixvec[seqstart:seqend+1]
                                                subfixvec2 = fixvec[seq2start:seq2end+1]
                                                subtup1 = rp.vecToTup(subfixvec1)
                                                subtup2 = rp.vecToTup(subfixvec2)
                                                if (subtup1[-1][0]=='MA')&(subtup2[0][0]=='MA'):
                                                    if subtup1[-1][1]>=subtup2[0][1]:
                                                        macounts[idx+1]=macounts[idx+1]-1
                                                    else:
                                                        macounts[idx] = macounts[idx]-1
                                    for idx,macnt in enumerate(macounts):
                                        if np.isnan(macnt):
                                            marates.append(np.nan)
                                        else:
                                            s = befafts[idx]
                                            marates.append(macnt/(len(s)*2.5)*60)
                                    #Spindles
                                    spcounts = []
                                    sprates = []
                                    for s in befafts:
                                        if len(s)==0:
                                            spcounts.append(np.nan)
                                        else:
                                            seqstart = s[0]
                                            seqend = s[-1]
                                            spcnt = 0
                                            for y in scaledSpindleTimes:
                                                spinstart = y[0]
                                                spinend = y[1]
                                                if (spinstart>=seqstart)&(spinend<=seqend):
                                                    spcnt+=1
                                            spcounts.append(spcnt)
                                    for idx,s in enumerate(befafts):
                                        if idx<len(befafts)-1:
                                            s2 = befafts[idx+1]
                                            if (len(s)>0)&(len(s2)>0):
                                                seqstart = s[0]
                                                seqend = s[-1]
                                                seq2start = s2[0]
                                                seq2end = s2[-1]
                                                for y in scaledSpindleTimes:
                                                    spinstart = y[0]
                                                    spinend = y[1]
                                                    if (spinstart<=seqend)&(spinend>=seq2start):
                                                        if (seqend-spinstart)>=(spinend-seq2start):
                                                            spcounts[idx] = spcounts[idx]+1
                                                        else:
                                                            spcounts[idx+1] = spcounts[idx+1]+1
                                    for idx,spcnt in enumerate(spcounts):
                                        if np.isnan(spcnt):
                                            sprates.append(np.nan)
                                        else:
                                            s = befafts[idx]
                                            sprates.append(spcnt/(len(s)*2.5)*60)
                                    allSigs2.append(Sigmas)
                                    allThes2.append(Thetas)
                                    allSpins2.append(sprates)
                                    allMAs2.append(marates)
            cnt2+=1
    print(rec)
                                    
#xvalues to plot                                    
befxes = np.arange(-3.5,0,1)                            
aftxes = np.arange(0.5,4,1)

#Sigma
fig5E1 = plt.figure()
plt.title('Sigma')
aSig = np.nanmean(allSigs2, axis=0) - 2.576*np.nanstd(allSigs2, axis=0)/np.sqrt(len(allSigs2))
bSig = np.nanmean(allSigs2, axis=0) + 2.576*np.nanstd(allSigs2, axis=0)/np.sqrt(len(allSigs2))
plt.plot(befxes, np.nanmean(allSigs2,axis=0)[0:4], color='k')
plt.plot(aftxes, np.nanmean(allSigs2,axis=0)[4:], color='k')
plt.scatter(befxes, np.nanmean(allSigs2,axis=0)[0:4], color='k', s=10)
plt.scatter(aftxes, np.nanmean(allSigs2,axis=0)[4:], color='k', s=10)
plt.fill_between(befxes,aSig[0:4],bSig[0:4], color='k', alpha=0.2)
plt.fill_between(aftxes,aSig[4:],bSig[4:], color='k', alpha=0.2)
sns.despine()
plt.ylim([600,1250])

#Theta
fig5E2 = plt.figure()
plt.title('Theta')
aThe = np.nanmean(allThes2, axis=0) - 2.576*np.nanstd(allThes2, axis=0)/np.sqrt(len(allThes2))
bThe = np.nanmean(allThes2, axis=0) + 2.576*np.nanstd(allThes2, axis=0)/np.sqrt(len(allThes2))
plt.plot(befxes, np.nanmean(allThes2,axis=0)[0:4], color='k')
plt.plot(aftxes, np.nanmean(allThes2,axis=0)[4:], color='k')
plt.scatter(befxes, np.nanmean(allThes2,axis=0)[0:4], color='k', s=10)
plt.scatter(aftxes, np.nanmean(allThes2,axis=0)[4:], color='k', s=10)
plt.fill_between(befxes,aThe[0:4],bThe[0:4], color='k', alpha=0.2)
plt.fill_between(aftxes,aThe[4:],bThe[4:], color='k', alpha=0.2)
sns.despine()
plt.ylim([1400,2200])

#Spindles
fig5E3 = plt.figure()
plt.title('Spindles')
aSpin = np.nanmean(allSpins2, axis=0) - 2.576*np.nanstd(allSpins2, axis=0)/np.sqrt(len(allSpins2))
bSpin = np.nanmean(allSpins2, axis=0) + 2.576*np.nanstd(allSpins2, axis=0)/np.sqrt(len(allSpins2))
plt.plot(befxes, np.nanmean(allSpins2,axis=0)[0:4], color='k')
plt.plot(aftxes, np.nanmean(allSpins2,axis=0)[4:], color='k')
plt.scatter(befxes, np.nanmean(allSpins2,axis=0)[0:4], color='k', s=10)
plt.scatter(aftxes, np.nanmean(allSpins2,axis=0)[4:], color='k', s=10)
plt.fill_between(befxes,aSpin[0:4],bSpin[0:4], color='k', alpha=0.2)
plt.fill_between(aftxes,aSpin[4:],bSpin[4:], color='k', alpha=0.2)
sns.despine()
plt.ylim([1.5,6])

#MAs
fig5E4 = plt.figure()
plt.title('MAs')
aMAs = np.nanmean(allMAs2, axis=0) - 2.576*np.nanstd(allMAs2, axis=0)/np.sqrt(len(allMAs2))
bMAs = np.nanmean(allMAs2, axis=0) + 2.576*np.nanstd(allMAs2, axis=0)/np.sqrt(len(allMAs2))
plt.plot(befxes, np.nanmean(allMAs2,axis=0)[0:4], color='k')
plt.plot(aftxes, np.nanmean(allMAs2,axis=0)[4:], color='k')
plt.scatter(befxes, np.nanmean(allMAs2,axis=0)[0:4], color='k', s=10)
plt.scatter(aftxes, np.nanmean(allMAs2,axis=0)[4:], color='k', s=10)
plt.fill_between(befxes,aMAs[0:4],bMAs[0:4], color='k', alpha=0.2)
plt.fill_between(aftxes,aMAs[4:],bMAs[4:], color='k', alpha=0.2)
sns.despine()
plt.ylim([0,1.5])

#Save figures
fig5E1.savefig(outpath+'f5e_sigmarel.pdf')                              
fig5E2.savefig(outpath+'f5e_thetarel.pdf')
fig5E3.savefig(outpath+'f5e_spindlerel.pdf')
fig5E4.savefig(outpath+'f5e_masrel.pdf')

plt.close(fig5E1)
plt.close(fig5E2)
plt.close(fig5E3)
plt.close(fig5E4)
