#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:31:32 2020

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

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/final_figures/3_SeqSingle/'

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

###############################################################################

###Figure 3A (LEFT) - Scatterplot of rem-nrem cycles divided by sequential and single

fig3A1 = plt.figure()
plt.scatter(nonDF.rem, nonDF.logsws, s=10, alpha=0.6, color='gray')
plt.scatter(seqDF.rem, seqDF.logsws, s=10, alpha=0.6, color=col2)
plt.scatter(sinDF.rem, sinDF.logsws, s=10, alpha=0.6, color=col1)
plt.plot(intersect_x, intersect_y, color='k')
plt.axvline(x=30, color='k',lw=1)
plt.xticks([0,60,120,180,240],["","","","",""])
plt.yticks([2,4,6,8],["","","",""])
sns.despine()
plt.ylim([2,9])

fig3A1.savefig(outpath+'f3A_scatter.png')


###Figure 3A (RIGHT) - Example of high and low distributions for REMpre = 30 s

#Calculate weights,means,st.devs with REMpre=30
rem = 30
k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
if k1>1:
    k1=1
k2 = 1-k1
m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
s2 = slow_lina*rem+slow_linb

#Low and High Gaussian distribution PDFs
f1 = lambda x: k1*stats.norm.pdf(x,m1,s1)
f2 = lambda x: k2*stats.norm.pdf(x,m2,s2)

#xvalues to plot
xplot = np.arange(2,9,0.01)

#Point where dividing line intersect with REMpre=30
intersectind = np.where(intersect_x==30)[0]
intersectval = intersect_y[intersectind][0][0]

#Plot
fig3A2 = plt.figure()
plt.plot(f1(xplot),xplot,color=col1,label='Single')
plt.plot(f2(xplot),xplot,color=col2,label='sequential')
plt.axhline(y=intersectval, color='k',lw=1,ls='--')
sns.despine()
plt.legend()
plt.yticks([2,4,6,8])
plt.ylim([2,9])

fig3A2.savefig(outpath+'f3A_expdf.pdf')

###############################################################################
###Figure 3B
#Pie Chart of Single and Sequential Percentages

labels = ['Single','Sequential']
sizes = [len(sinDF), len(seqDF)]
cols = [col1,col2]

fig3B = plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=cols)

fig3B.savefig(outpath+'f3B_pie.pdf')

###############################################################################

###Figure 3C
#Boxplot comparing REMpre of Sequential and Single cycles

boxcolors = [col2, col1]

fig3C,ax = plt.subplots()
bp = ax.boxplot([seqDF.rem, sinDF.rem], vert=True, patch_artist=True)
for idx,patch in enumerate(bp['boxes']):
    patch.set_facecolor(boxcolors[idx])
for patch in bp['medians']:
    plt.setp(patch, color='k')
sns.despine()
plt.xticks([1,2],['Sequential','Single'])

fig3C.savefig(outpath+'f3C_remprebox.pdf')


#Stats

stats.levene(seqDF.rem, sinDF.rem)
stats.ttest_ind(seqDF.rem,sinDF.rem,equal_var=False)
    
###############################################################################

###Figure 3D
#Histogram of |N| of sequential cycles

fig3D = plt.figure()
plt.hist(seqDF.sws, bins=20, color=col2)
plt.axvline(x=np.mean(seqDF.sws), color='k', ls='--', lw=1, label='mean')
sns.despine()
plt.legend()

fig3D.savefig(outpath+'f3D_seqswshist.pdf')

#Stats
np.mean(seqDF.sws)
np.std(seqDF.sws)

###############################################################################

###Figure 3E - Consecutive sequential REM-NREM cycles

#Reset indices of sequential cycles dataframe
sort_seqDF = seqDF.reset_index(drop=True)

#List to store indices of consecutive rows in dataframe
consecutive = []

#Go through sequential dataframe and find rows that are consecutive
for ind in range(len(sort_seqDF)-1):
    row1 = sort_seqDF.iloc[ind]
    row2 = sort_seqDF.iloc[ind+1]   
    rec1 = row1[5]
    rec2 = row2[5]   
    istart = row1[4]
    rem = row1[0]
    inter = row1[7]
    dur = (rem+inter)/2.5   
    istart2 = row2[4]
    
    if rec1==rec2:
        if istart2==istart+dur:
            consecutive.append(ind)

#List of tuples storing consecutive indices
consec_range = rp.ranges(consecutive)

#Count number of instances for 2 consecutive cycles, 3 consecutive cycles...
consec1 = 0
consec2 = 0
consec3 = 0
consec4 = 0

for x in consec_range:
    start = x[0]
    end = x[1]
    if end-start==0:
        consec1+=1
    if end-start==1:
        consec2+=1
    if end-start==2:
        consec3+=1
    if end-start==3:
        consec4+=1

consecsum = consec1*2 + consec2*3 + consec3*4 + consec4*5
singleseq = len(seqDF) - consecsum

#Convert counts to percentages
totalsum = len(seqDF)
singleper = float(singleseq)/totalsum*100
con1per = float(2*consec1)/totalsum*100
con2per = float(3*consec2)/totalsum*100
con3per = float(4*consec3)/totalsum*100
con4per = float(5*consec4)/totalsum*100

#Plot consecutive sequential cycle percentage
fig3E = plt.figure()
plt.bar([1,2,3,4,5],[singleper,con1per, con2per, con3per, con4per],0.4, color=col2)
sns.despine()

outpath = r'/home/cwlab08/Desktop/'

fig3E.savefig(outpath+'f3E_consecseq.pdf')


###############################################################################
###Figure 3F
#Spectral Density comparing REM and |N| of Sequential and Single cycles

##Parietal EEG
seqSwsSpec = []
sinSwsSpec = []
seqRemSpec = []
sinRemSpec = []

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
            #Find location of REM and |N| of cycles
            states = np.array([x[0] for x in sub])
            swsInds = np.where((states=='N')|(states=='MA'))[0]
            swsSeqs = rp.stateSeq(sub, swsInds)
            remInds = np.where((states=='R'))[0]
            remSeqs = rp.stateSeq(sub, remInds)
            #Calculate spectral density
            rem = sub[0][1]*2.5
            if rem>2.5: #REM periods need to be at least 5s
                subRemSpec = []
                subSwsSpec = []
                for s in remSeqs:
                    if len(s)*nbin>=nwin:
                        b = int((s[-1]+1)*nbin)
                        sup = list(range(int(s[0]*nbin), b))
                        if sup[-1]>len(EEG):
                            sup = list(range(int(s[0]*nbin), len(EEG)))
                        if len(sup) >= nwin:
                            Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                            subRemSpec.append(Pow)
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
                remSpec = np.array([x[ifreq] for x in subRemSpec])
                avgRemSpec = np.mean(remSpec,axis=0)
                
                sws = 0
                for x in sub:
                    if (x[0]=='N')or(x[0]=='MA'):
                        sws+=x[1]*2.5
                if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240]
                    if rp.isSequential(rem,sws,intersect_x,intersect_y): #Check if sequential or single
                        seqSwsSpec.append(avgSwsSpec)
                        seqRemSpec.append(avgRemSpec)
                    else:
                        sinSwsSpec.append(avgSwsSpec)
                        sinRemSpec.append(avgRemSpec)
            cnt2+=1
    print(rec)

#Limit inspection to frequencies in btw [0Hz,20Hz]                
ifreq = np.where(F<=20)
FN = F[ifreq]

seqRemS = np.array(seqRemSpec)
seqSwsS = np.array(seqSwsSpec)
sinRemS = np.array(sinRemSpec)
sinSwsS = np.array(sinSwsSpec)

#Calculate 99% confidence intervals
aseqRem = seqRemS.mean(axis=0) - 2.576*seqRemS.std(axis=0)/np.sqrt(len(seqRemS))
bseqRem = seqRemS.mean(axis=0) + 2.576*seqRemS.std(axis=0)/np.sqrt(len(seqRemS))
asinRem = sinRemS.mean(axis=0) - 2.576*sinRemS.std(axis=0)/np.sqrt(len(sinRemS))
bsinRem = sinRemS.mean(axis=0) + 2.576*sinRemS.std(axis=0)/np.sqrt(len(sinRemS))

aseqSws = seqSwsS.mean(axis=0) - 2.576*seqSwsS.std(axis=0)/np.sqrt(len(seqSwsS))
bseqSws = seqSwsS.mean(axis=0) + 2.576*seqSwsS.std(axis=0)/np.sqrt(len(seqSwsS))
asinSws = sinSwsS.mean(axis=0) - 2.576*sinSwsS.std(axis=0)/np.sqrt(len(sinSwsS))
bsinSws = sinSwsS.mean(axis=0) + 2.576*sinSwsS.std(axis=0)/np.sqrt(len(sinSwsS))


###Stats
#REM
eeg1RemFreq = []
eeg1Remwpval = [] #welch's test

#Perform welch's test for eaach frequency between [0,15]Hz
for i in range(46):
    curfreqind = i
    seqp = np.array([x[curfreqind] for x in seqRemS])
    sinp = np.array([x[curfreqind] for x in sinRemS])
    eeg1RemFreq.append(F[i])
    eeg1Remwpval.append(stats.ttest_ind(seqp,sinp,equal_var=False)[1])

#Store frequencies for which difference is significant at alpha=0.05
eeg1rem_sig1 = []
eeg1rem_sig2 = []
eeg1rem_sig3 = []

for idx,x in enumerate(eeg1Remwpval):
    if x<0.05:
        eeg1rem_sig1.append(eeg1RemFreq[idx])
    if x<0.01:
        eeg1rem_sig2.append(eeg1RemFreq[idx])
    if x<0.001:
        eeg1rem_sig3.append(eeg1RemFreq[idx])

#p<0.05
eeg1rem_sig1f1 = 3
eeg1rem_sig1f2 = np.arange(4,4.3333,0.0001)
eeg1rem_sig1f3 = np.arange(6.3333,6.6666,0.0001)
eeg1rem_sig1f4 = np.arange(7.3333,11, 0.0001)
eeg1rem_sig1f5 = np.arange(13,13.3333,0.0001)
#p<0.01
eeg1rem_sig2f1 = 4.3333
eeg1rem_sig2f2 = 6.6666
eeg1rem_sig2f3 = np.arange(7.3333,10.6666,0.0001)
#p<0.001
eeg1rem_sig3f1 = np.arange(7.3333,10.3333,0.0001)

#SWS
eeg1SWSFreq = []
eeg1SWSwpval = [] #welch's test

#Perform welch's test for eaach frequency between [0,15]Hz
for i in range(46):
    curfreqind = i
    seqp = np.array([x[curfreqind] for x in seqSwsS])
    sinp = np.array([x[curfreqind] for x in sinSwsS])
    eeg1SWSFreq.append(F[i])
    eeg1SWSwpval.append(stats.ttest_ind(seqp,sinp,equal_var=False)[1])

#Store frequencies for which difference is significant at alpha=0.05
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

#p<0.05
eeg1sws_sig1f1 = np.arange(0.3333,7.6666,0.0001)
eeg1sws_sig1f2 = np.arange(8.3333,15,0.0001)
#p<0.01
eeg1sws_sig2f1 = np.arange(0.3333,5.3333,0.0001)
eeg1sws_sig2f2 = np.arange(6,7.6666,0.0001)
eeg1sws_sig2f3 = np.arange(8.6666,15,0.0001)
#p<0.001
eeg1sws_sig3f1 = np.arange(0.6666,5.3333,0.0001)
eeg1sws_sig3f2 = np.arange(6.3333,7.6666,0.0001)
eeg1sws_sig3f3 = np.arange(9.3333,15,0.0001)



##### With averaged subsequences
fig3F1 = plt.figure()
plt.plot(FN, sinRemS.mean(axis=0), label='single', color=col1, lw=1)
plt.plot(FN, seqRemS.mean(axis=0), label='sequential', color=col2, lw=1)
plt.fill_between(FN, asinRem,bsinRem, color=col1, alpha=0.5)
plt.fill_between(FN, aseqRem,bseqRem, color=col2, alpha=0.5)
#pval<0.05
plt.scatter(eeg1rem_sig1f1,1400,color='k',s=1)
plt.plot(eeg1rem_sig1f2, np.repeat(1400, len(eeg1rem_sig1f2)),lw=1,color='k')
plt.plot(eeg1rem_sig1f3, np.repeat(1400, len(eeg1rem_sig1f3)),lw=1,color='k')
plt.plot(eeg1rem_sig1f4, np.repeat(1400, len(eeg1rem_sig1f4)),lw=1,color='k')
plt.plot(eeg1rem_sig1f5, np.repeat(1400, len(eeg1rem_sig1f5)),lw=1,color='k')
#pval<0.01
plt.scatter(eeg1rem_sig2f1,1500,color='k',s=1)
plt.scatter(eeg1rem_sig2f2,1500,color='k',s=1)
plt.plot(eeg1rem_sig2f3, np.repeat(1500, len(eeg1rem_sig2f3)),lw=1,color='k')
#pval<0.001
plt.plot(eeg1rem_sig3f1, np.repeat(1600, len(eeg1rem_sig3f1)),lw=1,color='k')
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,1650])


fig3F2 = plt.figure()
plt.plot(FN, seqSwsS.mean(axis=0), label='Sequential', color=col2, lw=1)
plt.plot(FN, sinSwsS.mean(axis=0), label='Single', color=col1, lw=1)
plt.fill_between(FN, asinSws,bsinSws, color=col1, alpha=0.5)
plt.fill_between(FN, aseqSws,bseqSws, color=col2, alpha=0.5)
#pval<0.05
plt.plot(eeg1sws_sig1f1, np.repeat(1050,len(eeg1sws_sig1f1)),lw=1,color='k')
plt.plot(eeg1sws_sig1f2, np.repeat(1050,len(eeg1sws_sig1f2)),lw=1,color='k')
#pval<0.001
plt.plot(eeg1sws_sig2f1, np.repeat(1150,len(eeg1sws_sig2f1)),lw=1,color='k')
plt.plot(eeg1sws_sig2f2, np.repeat(1150,len(eeg1sws_sig2f2)),lw=1,color='k')
plt.plot(eeg1sws_sig2f3, np.repeat(1150,len(eeg1sws_sig2f3)),lw=1,color='k')
#pval<0.001
plt.plot(eeg1sws_sig3f1, np.repeat(1250,len(eeg1sws_sig3f1)),lw=1,color='k')
plt.plot(eeg1sws_sig3f2, np.repeat(1250,len(eeg1sws_sig3f2)),lw=1,color='k')
plt.plot(eeg1sws_sig3f3, np.repeat(1250,len(eeg1sws_sig3f3)),lw=1,color='k')
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,1300])


fig3F1.savefig(outpath+'f3f1_v2.pdf')
fig3F2.savefig(outpath+'f3f2_v2.pdf')

##Prefrontal EEG


tseqSwsSpec2 = []
tsinSwsSpec2 = []
tseqRemSpec2 = []
tsinRemSpec2 = []

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
            fixtup = rp.ma_thr(Mtup, 8)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 8)
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
            remInds = np.where((states=='R'))[0]
            remSeqs = rp.stateSeq(sub, remInds)
            #Calculate Spectral density
            rem = sub[0][1]*2.5
            if rem>2.5:
                remSpec = []
                swsSpec = []
                subSwsSpec = []
                subRemSpec = []
                for s in remSeqs:
                    if len(s)*nbin>=nwin:
                        b = int((s[-1]+1)*nbin)
                        sup = list(range(int(s[0]*nbin), b))
                        if sup[-1]>len(EEG):
                            sup = list(range(int(s[0]*nbin), len(EEG)))
                        if len(sup) >= nwin:
                            Pow,F = rp.power_spectrum(EEG[sup], nwin, 1/sr)
                            remSpec.append(Pow)
                            subRemSpec.append(Pow)
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
                tremSpec = np.array([x[ifreq] for x in subRemSpec])
                avgremSpec = np.mean(tremSpec,axis=0)
                
                sws = 0
                for x in sub:
                    if (x[0]=='N')or(x[0]=='MA'):
                        sws+=x[1]*2.5
                if (rem>=7.5)&(rem<240): #Limit to cycles with REMpre in [7.5,240]
                    if rp.isSequential(rem,sws,intersect_x,intersect_y): #Check if cycle is sequential or single
                        tseqSwsSpec2.append(avgswsSpec)
                        tseqRemSpec2.append(avgremSpec)
                    else:
                        tsinSwsSpec2.append(avgswsSpec)
                        tsinRemSpec2.append(avgremSpec)
            cnt2+=1
    print(rec)

#Limit inspection to frequencies in btw [0Hz,20Hz]                   
ifreq = np.where(F<=20)
FN = F[ifreq]

seqRemS2 = np.array(tseqRemSpec2)
seqSwsS2 = np.array(tseqSwsSpec2)
sinRemS2 = np.array(tsinRemSpec2)
sinSwsS2 = np.array(tsinSwsSpec2)

#Calculate 99% confidence intervals
aseqRem2 = seqRemS2.mean(axis=0) - 2.576*seqRemS2.std(axis=0)/np.sqrt(len(seqRemS2))
bseqRem2 = seqRemS2.mean(axis=0) + 2.576*seqRemS2.std(axis=0)/np.sqrt(len(seqRemS2))
asinRem2 = sinRemS2.mean(axis=0) - 2.576*sinRemS2.std(axis=0)/np.sqrt(len(sinRemS2))
bsinRem2 = sinRemS2.mean(axis=0) + 2.576*sinRemS2.std(axis=0)/np.sqrt(len(sinRemS2))

aseqSws2 = seqSwsS2.mean(axis=0) - 2.576*seqSwsS2.std(axis=0)/np.sqrt(len(seqSwsS2))
bseqSws2 = seqSwsS2.mean(axis=0) + 2.576*seqSwsS2.std(axis=0)/np.sqrt(len(seqSwsS2))
asinSws2 = sinSwsS2.mean(axis=0) - 2.576*sinSwsS2.std(axis=0)/np.sqrt(len(sinSwsS2))
bsinSws2 = sinSwsS2.mean(axis=0) + 2.576*sinSwsS2.std(axis=0)/np.sqrt(len(sinSwsS2))


###Stats
#REM
eeg2RemFreq = []
eeg2Remwpval = [] #welch's test

for i in range(46):
    curfreqind = i
    seqp = np.array([x[curfreqind] for x in tseqRemSpec2])
    sinp = np.array([x[curfreqind] for x in tsinRemSpec2])
    eeg2RemFreq.append(F[i])
    eeg2Remwpval.append(stats.ttest_ind(seqp,sinp,equal_var=False)[1])

eeg2rem_sig1 = []
eeg2rem_sig2 = []
eeg2rem_sig3 = []

for idx,x in enumerate(eeg2Remwpval):
    if x<0.05:
        eeg2rem_sig1.append(eeg2RemFreq[idx])
    if x<0.01:
        eeg2rem_sig2.append(eeg2RemFreq[idx])
    if x<0.001:
        eeg2rem_sig3.append(eeg2RemFreq[idx])

#pval<0.05
eeg2rem_sig1f1 = np.arange(1,7,0.0001)
eeg2rem_sig1f2 = 7.6666
eeg2rem_sig1f3 = np.arange(9,15,0.0001)
#pval<0.01
eeg2rem_sig2f1 = np.arange(1,7,0.0001)
eeg2rem_sig2f2 = 7.6666
eeg2rem_sig2f3 = np.arange(9,15,0.0001)
#pval<0.001
eeg2rem_sig3f1 = np.arange(1.3333,6.6666,0.0001)
eeg2rem_sig3f2 = np.arange(9,15,0.0001)


#SWS
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

#pval<0.05
eeg2sws_sig1f1 = np.arange(0.3333,6,0.0001)
eeg2sws_sig1f2 = np.arange(7,15,0.0001)
#pval<0.01
eeg2sws_sig2f1 = np.arange(0.6666,6,0.0001)
eeg2sws_sig2f2 = np.arange(7.3333,15,0.0001)
#pval<0.001
eeg2sws_sig3f1 = np.arange(0.6666,6,0.0001)
eeg2sws_sig3f2 = np.arange(7.3333,15,0.0001)



#### Using average of subsequences
fig3F3 = plt.figure()
plt.plot(FN, sinRemS2.mean(axis=0), label='single', color=col1, lw=1)
plt.plot(FN, seqRemS2.mean(axis=0), label='sequential', color=col2, lw=1)
plt.fill_between(FN, aseqRem2, bseqRem2, color=col2, alpha=0.5)
plt.fill_between(FN, asinRem2, bsinRem2, color=col1, alpha=0.5)
#pval<0.05
plt.plot(eeg2rem_sig1f1, np.repeat(525, len(eeg2rem_sig1f1)),lw=1,color='k')
plt.scatter(eeg2rem_sig1f2, 525, s=1, color='k')
plt.plot(eeg2rem_sig1f3, np.repeat(525, len(eeg2rem_sig1f3)),lw=1,color='k')
#pval<0.01
plt.plot(eeg2rem_sig2f1, np.repeat(600, len(eeg2rem_sig2f1)),lw=1,color='k')
plt.scatter(eeg2rem_sig2f2, 600, s=1, color='k')
plt.plot(eeg2rem_sig2f3, np.repeat(600, len(eeg2rem_sig2f3)),lw=1,color='k')
#pval<0.001
plt.plot(eeg2rem_sig3f1, np.repeat(675, len(eeg2rem_sig3f1)),lw=1,color='k')
plt.plot(eeg2rem_sig3f2, np.repeat(675, len(eeg2rem_sig3f2)),lw=1,color='k')
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,700])


fig3F4 = plt.figure()
plt.plot(FN, seqSwsS2.mean(axis=0), label='Sequential', color=col2, lw=1)
plt.plot(FN, sinSwsS2.mean(axis=0), label='Single', color=col1, lw=1)
plt.fill_between(FN, aseqSws2, bseqSws2, color=col2, alpha=0.5)
plt.fill_between(FN, asinSws2, bsinSws2, color=col1, alpha=0.5)
#pval<0.05
plt.plot(eeg2sws_sig1f1, np.repeat(1000, len(eeg2sws_sig1f1)),lw=1,color='k')
plt.plot(eeg2sws_sig1f2, np.repeat(1000, len(eeg2sws_sig1f2)),lw=1,color='k')
#pval<0.01
plt.plot(eeg2sws_sig2f1, np.repeat(1100, len(eeg2sws_sig2f1)),lw=1,color='k')
plt.plot(eeg2sws_sig2f2, np.repeat(1100, len(eeg2sws_sig2f2)),lw=1,color='k')
#pval<0.001
plt.plot(eeg2sws_sig3f1, np.repeat(1200, len(eeg2sws_sig3f1)),lw=1,color='k')
plt.plot(eeg2sws_sig3f2, np.repeat(1200, len(eeg2sws_sig3f2)),lw=1,color='k')
sns.despine()
plt.legend()
plt.xticks([0,5,10,15])
plt.ylim([0,1225])


fig3F3.savefig(outpath+'f3f3_v2.pdf')
fig3F4.savefig(outpath+'f3f4_v2.pdf')

