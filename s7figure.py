#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 21:16:20 2021

@author: cwlab08
"""


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

ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

outpath = r'/home/cwlab08/Desktop/final_figures/Sfig3/'
current_palette = sns.color_palette('muted', 10)
#sns.palplot(current_palette)

col1 = current_palette[3]
col2 = current_palette[0]


remDF = rp.standard_recToDF(ppath, recordings, 8, False)
remDF['loginter'] = remDF['inter'].apply(np.log)
remDF = remDF.loc[remDF.rem<240]

###############################################################################

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
shigh_loga = logfitDF.shigh[0]
shigh_logb = logfitDF.shigh[1]
shigh_logc = logfitDF.shigh[2]

slow_lina = linfitDF.slow[0]
slow_linb = linfitDF.slow[1]
shigh_lina = linfitDF.shigh[0]
shigh_linb = linfitDF.shigh[1]

###############################################################################

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

###############################################################################

remDF = remDF.loc[remDF.rem<240]
nonDF = remDF.loc[remDF.rem<7.5]
subDF = remDF.loc[remDF.rem>=7.5]

subDF = subDF.reset_index(drop=True)

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

###############################################################################
recordings2 = [x for x in recordings if x!='HA41_081618n1']

remSpec30 = []
remSpec60 = []
remSpec90 = []
remSpec120 = []
remSpec150 = []
remSpec180 = []
remSpec210 = []
remSpec240 = []

twin = 3

for rec in recordings2:
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    sr = rp.get_snr(ppath, rec)
    nbin = int(np.round(sr)*2.5)
    dt = nbin*1/sr
    nwin = np.round(twin*sr)
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'))['EEG2']).astype('float')
    #EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'))['EEG2']).astype('float')
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
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240):
                if rem<30:
                    remSpec30.extend(remSpec)
                elif rem<60:
                    remSpec60.extend(remSpec)
                elif rem<90:
                    remSpec90.extend(remSpec)
                elif rem<120:
                    remSpec120.extend(remSpec)
                elif rem<150:
                    remSpec150.extend(remSpec)
                elif rem<180:
                    remSpec180.extend(remSpec)
                elif rem<210:
                    remSpec210.extend(remSpec)
                else:
                    remSpec240.extend(remSpec)                    
            cnt2+=1
    print(rec)

df = F[1]-F[0]
ifreq2 = np.where(F<=15)
F2 = F[ifreq2]

seqSplitDFs = rp.splitData(seqDF,30)
sinSplitDFs = rp.splitData(sinDF,30)

cSeqr30 = len(seqSplitDFs[0])/len(seqDF)
cSeqr60 = len(seqSplitDFs[1])/len(seqDF)
cSeqr90 = len(seqSplitDFs[2])/len(seqDF)
cSeqr120 = len(seqSplitDFs[3])/len(seqDF)
cSeqr150 = len(seqSplitDFs[4])/len(seqDF)

cSinr30 = len(sinSplitDFs[0])/len(sinDF)
cSinr60 = len(sinSplitDFs[1])/len(sinDF)
cSinr90 = len(sinSplitDFs[2])/len(sinDF)
cSinr120 = len(sinSplitDFs[3])/len(sinDF)
cSinr150 = len(sinSplitDFs[4])/len(sinDF)
cSinr180 = len(sinSplitDFs[5])/len(sinDF)
cSinr210 = len(sinSplitDFs[6])/len(sinDF)
cSinr240 = len(sinSplitDFs[7])/len(sinDF)

sub30 = np.array([x[ifreq2] for x in remSpec30])
sub60 = np.array([x[ifreq2] for x in remSpec60])
sub90 = np.array([x[ifreq2] for x in remSpec90])
sub120 = np.array([x[ifreq2] for x in remSpec120])
sub150 = np.array([x[ifreq2] for x in remSpec150])
sub180 = np.array([x[ifreq2] for x in remSpec180])
sub210 = np.array([x[ifreq2] for x in remSpec210])
sub240 = np.array([x[ifreq2] for x in remSpec240])

subbylen = [sub30,sub60,sub90,sub120,sub150,sub180,sub210,sub240]

sfig2A = plt.figure()
cNorm = colors.Normalize(vmin=0, vmax=17)
cmap = plt.get_cmap('YlOrRd')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for idx,sub in enumerate(subbylen):
    colorVal = scalarMap.to_rgba(2*idx+1)
    plt.plot(F2, sub.mean(axis=0), color=colorVal, label=str(30*idx)+'<=rem<'+str(30*(idx+1)))
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,600])    

#sfig2A.savefig(outpath+'sf2A_remspecbydur.pdf')

###############################################################################


seqRemSpec2 = []
sinRemSpec2 = []

twin=3

recordings2 = [x for x in recordings if x!='HA41_081618n1']

for rec in recordings2:
    recs, nswitch, start = rp.find_light(ppath, rec, False)
    sr = rp.get_snr(ppath, rec)
    nbin = int(np.round(sr)*2.5)
    dt = nbin*1/sr
    nwin = np.round(twin*sr)
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'))['EEG2']).astype('float')
    #EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'))['EEG2']).astype('float')
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
            sws = 0
            for x in sub:
                if (x[0]=='N')or(x[0]=='MA'):
                    sws+=x[1]*2.5
            if (rem>=7.5)&(rem<240):
                if rp.isSequential(rem,sws,intersect_x,intersect_y):
                    seqRemSpec2.extend(remSpec)
                else:
                    sinRemSpec2.extend(remSpec)
            cnt2+=1
    print(rec)
                
ifreq = np.where(F<=15)
FN = F[ifreq]

df = F[1]-F[0]

seqRemS2 = np.array([x[ifreq] for x in seqRemSpec2])
sinRemS2 = np.array([x[ifreq] for x in sinRemSpec2])

aseqRem = seqRemS2.mean(axis=0) - 2.576*seqRemS2.std(axis=0)/np.sqrt(len(seqRemS2))
bseqRem = seqRemS2.mean(axis=0) + 2.576*seqRemS2.std(axis=0)/np.sqrt(len(seqRemS2))

sfig2B1 = plt.figure()
plt.title('Sequential')
plt.plot(F2, cSeqr30*sub30.mean(axis=0)+cSeqr60*sub60.mean(axis=0)+cSeqr90*sub90.mean(axis=0)+cSeqr120*sub120.mean(axis=0)+cSeqr150*sub150.mean(axis=0), color = col2, label='Weighted average', ls='--')
plt.plot(FN, seqRemS2.mean(axis=0), label='Actual', color=col2)
plt.fill_between(FN, aseqRem, bseqRem, color=col2, alpha=0.4)
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,600])

#sfig2B1.savefig(outpath+'sf2B_sequential.pdf')

asinRem = sinRemS2.mean(axis=0) - 2.576*sinRemS2.std(axis=0)/np.sqrt(len(sinRemS2))
bsinRem = sinRemS2.mean(axis=0) + 2.576*sinRemS2.std(axis=0)/np.sqrt(len(sinRemS2))

sfig2B2 = plt.figure()
plt.title('Single')
plt.plot(F2, cSinr30*sub30.mean(axis=0)+cSinr60*sub60.mean(axis=0)+cSinr90*sub90.mean(axis=0)+cSinr120*sub120.mean(axis=0)+cSinr150*sub150.mean(axis=0)+cSinr180*sub180.mean(axis=0)+cSinr210*sub210.mean(axis=0)+cSinr240*sub240.mean(axis=0), color=col1, label='Weighted average', ls='--')
plt.plot(FN, sinRemS2.mean(axis=0), color=col1, label='Actual')
plt.fill_between(FN,asinRem,bsinRem, color=col1, alpha=0.4)
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,600])

#sfig2B2.savefig(outpath+'sf2B_single.pdf')

##############################################################################