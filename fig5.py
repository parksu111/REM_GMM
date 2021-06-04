#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:26:30 2021

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
import pingouin

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/final_figures/5_WakeRole/'

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
### Fig 5A - Sequential Cycles with wake and without wake

#Divide sequential cycles to those with wake and without wake
seqwwDF = seqDF.loc[seqDF.wake>0]
seqwoDF = seqDF.loc[seqDF.wake==0]

#Lables/colors and values for boxplot
labels = ['|W|=0','|W|>0']
sizes1 = [len(seqwoDF), len(seqwwDF)]
cols1 = ['skyblue',col2]

#Pie Chart
fig5A1 = plt.figure()
plt.pie(sizes1, labels=labels, autopct='%1.1f%%', colors=cols1)

#Boxplot to compare |N|
fig5A2,ax = plt.subplots(figsize=[4,4.8])
bp = ax.boxplot([seqwoDF.sws,seqwwDF.sws],positions=[0.3,0.6],widths=0.15, vert=True, patch_artist=True)
for ind,patch in enumerate(bp['boxes']):
    patch.set_facecolor(cols1[ind])
sns.despine()
plt.xticks([0.3,0.6],labels)
plt.xlim([0.15,0.75])

#Statistics
stats.levene(seqwoDF.sws,seqwwDF.sws)
stats.ttest_ind(seqwoDF.sws,seqwwDF.sws,equal_var=True)

#Save figures
fig5A1.savefig(outpath+'f5a_seqpiechart.pdf')
fig5A2.savefig(outpath+'f5a_seqboxplot.pdf')


###############################################################################
### Fig 5B - Single Cycles with and without wake

#Divide single cycles to those with wake and without wake
sinwwDF = sinDF.loc[sinDF.wake>0]
sinwoDF = sinDF.loc[sinDF.wake==0]

#Colors and values for boxplot
sizes2 = [len(sinwoDF), len(sinwwDF)]
cols2 = ['mistyrose',col1]

#Pie Chart
fig5B1 = plt.figure()
plt.pie(sizes2, labels=labels, autopct='%1.1f%%', colors=cols2)

#Boxplot to compare |N|
fig5B2,ax = plt.subplots(figsize=[4,4.8])
bp = ax.boxplot([sinwoDF.sws,sinwwDF.sws],positions=[0.3,0.6],widths=0.15, vert=True, patch_artist=True)
for ind,patch in enumerate(bp['boxes']):
    patch.set_facecolor(cols2[ind])
sns.despine()
plt.xticks([0.3,0.6],labels)
plt.xlim([0.15,0.75])

#Statistics
stats.levene(sinwoDF.sws,sinwwDF.sws)
stats.ttest_ind(sinwoDF.sws,sinwwDF.sws,equal_var=False)

#Save figures
fig5B1.savefig(outpath+'f5b_sinpiechart.pdf')
fig5B2.savefig(outpath+'f5b_sinboxplot.pdf')


###############################################################################
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
            fixtup = rp.ma_thr(Mtup, 8)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
            fixtup = rp.ma_thr(Mtup, 8)
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

#Percentage of cycles with MANY (>8) wake epsidoes
(wcnt9+wcnt10+wcnt11+wcnt12+wcnt13)/totalcnt
(wcnt6 + wcnt7 + wcnt8 +wcnt9+wcnt10+wcnt11+wcnt12+wcnt13)/totalcnt

###############################################################################
####Fig 5D - |N| by |W| and #wake blocks

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



###stats
pingouin.homoscedasticity(data=wakeDF,dv="sws",group="wpGroup")
pingouin.welch_anova(data=wakeDF,dv="sws",between="wpGroup")

pingouin.homoscedasticity(data=subwcDF, dv="sws", group="wcount")
pingouin.welch_anova(data=subwcDF, dv="sws", between="wcount")


#Save figures
fig5D1.savefig(outpath+'f5d_swsbywake.pdf')
fig5D2b.savefig(outpath+'f5d_swsbywcount2.pdf')



###############################################################################
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

#Save figures
fig5E1.savefig(outpath+'f5e_sigmarel.pdf')                              
fig5E2.savefig(outpath+'f5e_thetarel.pdf')
fig5E3.savefig(outpath+'f5e_spindlerel.pdf')
fig5E4.savefig(outpath+'f5e_masrel.pdf')


###############################################################################
#### Fig5F - Drop in power, spindle rate, MA rate in by wake

#Lists to store results
allSigs = []
allSpins = []
allMAs = []
allWakes = []
allThes = []

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
                                if (len(befseq)>24)&(len(aftseq)>24): #Check if sequences before and after are long enough
                                    #Find indices for sequences before and after
                                    for i in range(1):
                                        start = 24*i
                                        end = 24*(i+1)
                                        if len(revbef)>=end:
                                            subbef = revbef[start:end][::-1]
                                            befses.append(subbef)
                                        else:
                                            befses.append([])
                                    splitlen = int(len(aftseq)/24)
                                    for i in range(splitlen):
                                        start = 24*i
                                        end = 24*(i+1)
                                        if len(aftseq)>=end:
                                            subaft = aftseq[start:end]
                                            aftses.append(subaft)
                                        else:
                                            aftses.append([])
                                    befafts = []
                                    befafts.extend(befses[::-1])
                                    befafts.extend(aftses)
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
                                    allSigs.append(Sigmas)
                                    allThes.append(Thetas)
                                    allSpins.append(sprates)
                                    allMAs.append(marates)
                                    allWakes.append(curwake)
            cnt2+=1
    print(rec)

# Calculate drop/increase in quantities and time it takes to reach original level
sigInd = []
theInd = []
spinInd = []
maInd = []
sigDrop = []
theDrop = []
spinDrop = []
maDrop = []
passWake = []

for i in range(len(allSigs)):
    cursig = allSigs[i]
    curthe = allThes[i]
    curspin = allSpins[i]
    curma = allMAs[i]
    befsig = cursig[0]
    befthe = curthe[0]
    befspin = curspin[0]
    befma = curma[0]
    sigcnt = -1 
    thecnt = -1
    spincnt = -1
    macnt = -1
    for idx,x in enumerate(cursig):
        if idx>0:
            if x>=befsig:
                sigcnt=idx
                break
    for idx,x in enumerate(curthe):
        if idx>0:
            if x>=befthe:
                thecnt=idx
                break
    for idx,x in enumerate(curspin):
        if idx>0:
            if x>=befspin:
                spincnt=idx
                break
    for idx,x in enumerate(curma):
        if idx>0:
            if x<=befma:
                macnt=idx
                break
    sigInd.append(sigcnt)
    theInd.append(thecnt)
    spinInd.append(spincnt)
    maInd.append(macnt)
    sigDrop.append(cursig[0]-cursig[1])
    theDrop.append(curthe[0]-curthe[1])
    spinDrop.append(curspin[0]-curspin[1])
    maDrop.append(curma[1]-curma[0])
    passWake.append(allWakes[i])


#Compile into dataframe
passDF = pd.DataFrame(list(zip(passWake,sigDrop,theDrop,spinDrop,maDrop)),columns=['wake','sigd','thed','spind','madr'])

wperc20 = np.percentile(allWakes,20)
wperc40 = np.percentile(allWakes,40)
wperc60 = np.percentile(allWakes,60)
wperc80 = np.percentile(allWakes,80)


#Divide into groups by percentile of |W|
wpgroup = []
for x in passDF.wake:
    if x<wperc20:
        wpgroup.append(20)
    elif x<wperc40:
        wpgroup.append(40)
    elif x<wperc60:
        wpgroup.append(60)
    elif x<wperc80:
        wpgroup.append(80)
    else:
        wpgroup.append(100)

passDF['wpgroup'] = wpgroup



fig5F1 = plt.figure()
sns.barplot(x="wpgroup",y="sigd",data=passDF,palette=sns.color_palette("YlOrRd"))
sns.despine()

fig5F2 = plt.figure()
sns.barplot(x="wpgroup",y="thed",data=passDF,palette=sns.color_palette("YlOrRd"))
sns.despine()

fig5F3 = plt.figure()
sns.barplot(x="wpgroup",y="spind",data=passDF,palette=sns.color_palette("YlOrRd"))
sns.despine()

fig5F4 = plt.figure()
sns.barplot(x="wpgroup",y="madr",data=passDF,palette=sns.color_palette("YlOrRd"))
sns.despine()


#Save figure
fig5F1.savefig(outpath+'f5f_sigmadrop.pdf')
fig5F2.savefig(outpath+'f5f_thetadrop.pdf')
fig5F3.savefig(outpath+'f5f_spindrop.pdf')
fig5F4.savefig(outpath+'f5f_madrop.pdf')


###Stats
#sigma
pingouin.homoscedasticity(data=passDF,dv="sigd",group="wpgroup")
pingouin.anova(data=passDF,dv="sigd",between="wpgroup")

#theta
pingouin.homoscedasticity(data=passDF,dv="thed",group="wpgroup")
pingouin.welch_anova(data=passDF,dv="thed",between="wpgroup")

#spindles
pingouin.homoscedasticity(data=passDF,dv="spind",group="wpgroup")
pingouin.anova(data=passDF,dv="spind",between="wpgroup")

#MAs
pingouin.homoscedasticity(data=passDF,dv="madr",group="wpgroup")
pingouin.welch_anova(data=passDF,dv="madr",between="wpgroup")





