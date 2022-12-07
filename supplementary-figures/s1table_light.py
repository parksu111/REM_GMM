#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:22:38 2021

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
import statsmodels.api as sm


#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/revision_figures/light_individual/'

#Define colors
current_palette = sns.color_palette('muted', 10)
col1 = current_palette[3]
col2 = current_palette[0]

#Color palette for light and dark phase
tpal = ['mediumslateblue','darkslateblue']

#Make dataframe containing all REM-NREM cycles for light and dark phases
lightDF = rp.standard_recToDF(ppath, recordings, 8, False)
darkDF = rp.standard_recToDF(ppath, recordings, 8, True)

#Read in gmm parameters for light pahse
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


# Find intersection between low and high gaussian distributions (light phase)
intersection = []

for rem in np.arange(7.5,240,2.5):
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




#subset light phase dataframe
lightDF = lightDF.loc[lightDF.rem<240]
nonDF = lightDF.loc[lightDF.rem<7.5]
subDF = lightDF.loc[lightDF.rem>=7.5]
subDF = subDF.reset_index(drop=True)

#Divide light data into sequential and single cycles
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

#subset light phase dataframe
remDF = rp.standard_recToDF(ppath, recordings, 8, False)
rem_recordings = remDF.recording
rem_mice = [x.split('_')[0] for x in rem_recordings]
remDF['mname'] = rem_mice


mmouse = list(set(remDF.mname))
mrems = []
msws = []
mwakes = []
minters = []

mcorr_rpresws = []
mcorr_rprewake = []
mcorr_rpreinter = []

mpercSin = []
mpercSeq = []

for mouse in mmouse:
    subDF = remDF.loc[remDF.mname==mouse]
    mean_rem = np.mean(subDF.rem)
    mean_sws = np.mean(subDF.sws)
    mean_wake = np.mean(subDF.wake)
    mean_inter = np.mean(subDF.inter)
    mrems.append(mean_rem)
    msws.append(mean_sws)
    mwakes.append(mean_wake)
    minters.append(mean_inter)
    subDF = subDF.reset_index(drop=True)
    rems = subDF.rem
    swss = subDF.sws
    wakes = subDF.wake
    inters = subDF.inter
    totcnt = 0
    sincnt = 0
    seqcnt = 0
    for idx,x in enumerate(rems):
        sws = swss[idx]
        if (rem>=7.5)&(rem<240):
            totcnt+=1
            if rp.isSequential(rem,sws,intersect_x,intersect_y):
                seqcnt+=1
            else:
                sincnt+=1
    mpercSin.append(float(sincnt)/totcnt*100)
    mpercSeq.append(float(seqcnt)/totcnt*100)
    #rem vs nrem
    x1 = rems
    y1 = swss
    x1 = sm.add_constant(x1)
    mod1 = sm.OLS(y1,x1)
    res1 = mod1.fit()
    mcorr_rpresws.append(res1.rsquared)
    #rem vs wake
    x2 = rems
    y2 = wakes
    x2 = sm.add_constant(x2)
    mod2 = sm.OLS(y2,x2)
    res2 = mod2.fit()
    mcorr_rprewake.append(res2.rsquared)
    #rem vs inter
    x3 = rems
    y3 = inters
    x3 = sm.add_constant(x3)
    mod3 = sm.OLS(y3,x3)
    res3 = mod3.fit()
    mcorr_rpreinter.append(res3.rsquared)

## State mean duration
all_means = []
all_states = []
for x in mrems:
    all_means.append(x)
    all_states.append('rem')
for x in msws:
    all_means.append(x)
    all_states.append('nrem')
for x in mwakes:
    all_means.append(x)
    all_states.append('wake')
for x in minters:
    all_means.append(x)
    all_states.append('inter')

stateDF = pd.DataFrame(list(zip(all_means, all_states)),columns=['means','state'])

fig1 = plt.figure()
sns.barplot(x="state",y="means",data=stateDF, palette=['cyan','gray','purple','blue'])
sns.swarmplot(x="state", y="means",data=stateDF,color='k',size=3)
sns.despine()

fig1.savefig(outpath+'stateperc.pdf')

#stats
mean_states = [np.mean(mrems), np.mean(msws), np.mean(mwakes), np.mean(minters)]
std_states = [np.std(mrems), np.std(msws), np.std(mwakes), np.std(minters)]



#### Percent of sequential and single cycles
ssperc_percs = []
ssperc_states = []
for x in mpercSeq:
    ssperc_percs.append(x)
    ssperc_states.append('sequential')
for x in mpercSin:
    ssperc_percs.append(x)
    ssperc_states.append('single')


sspercDF = pd.DataFrame(list(zip(ssperc_percs, ssperc_states)),columns=['perc','state'])

fig2 = plt.figure()
sns.barplot(x="state",y="perc",data=sspercDF, palette=[col2,col1])
sns.swarmplot(x="state",y="perc",data=sspercDF,color='k',size=3)
sns.despine()

fig2.savefig(outpath+'seqsinperc.pdf')

#stats
mean_percs = [np.mean(mpercSeq), np.mean(mpercSin)]
std_percs = [np.std(mpercSeq), np.std(mpercSin)]


#### REMpre vs correlations
all_corrs = []
mcorr_states = []

for x in mcorr_rpresws:
    all_corrs.append(x)
    mcorr_states.append('sws')
for x in mcorr_rprewake:
    all_corrs.append(x)
    mcorr_states.append('wake')
for x in mcorr_rpreinter:
    all_corrs.append(x)
    mcorr_states.append('inter')

corrDF = pd.DataFrame(list(zip(all_corrs,mcorr_states)),columns=['corr', 'state'])

fig3 = plt.figure()
sns.barplot(x="state",y="corr", data=corrDF, palette=['gray','purple','blue'])
sns.swarmplot(x="state",y="corr", data=corrDF, color='k',size=3)
sns.despine()
plt.ylabel('R^2')
    
fig3.savefig(outpath+'rpresws_corr.pdf')

#stats
mean_corrs = [np.mean(mcorr_rpresws), np.mean(mcorr_rprewake), np.mean(mcorr_rpreinter)]
std_corrs = [np.std(mcorr_rpresws), np.std(mcorr_rprewake), np.std(mcorr_rpreinter)]

################# avg duration of single and sequential episodes


seq_recordings = seqDF.recording
sin_recordings = sinDF.recording

seq_mice = [x.split('_')[0] for x in seq_recordings]
sin_mice = [x.split('_')[0] for x in sin_recordings]

seqDF['mname'] = seq_mice
sinDF['mname'] = sin_mice

seq_mmouse = list(set(seqDF.mname))
sin_mmouse = list(set(sinDF.mname))

seq_remmeans = []
for mouse in seq_mmouse:
    subDF = seqDF.loc[seqDF.mname==mouse]
    seq_remmeans.append(np.mean(subDF.rem))

sin_remmeans = []
for mouse in sin_mmouse:
    subDF = sinDF.loc[sinDF.mname==mouse]
    sin_remmeans.append(np.mean(subDF.rem))

seqsin_remmeans = []
seqsin_states = []
for x in seq_remmeans:
    seqsin_remmeans.append(x)
    seqsin_states.append('sequential')
for x in sin_remmeans:
    seqsin_remmeans.append(x)
    seqsin_states.append('single')

seqsinDF = pd.DataFrame(list(zip(seqsin_remmeans,seqsin_states)),columns=['rem','state'])

fig4 = plt.figure()
sns.barplot(x="state", y="rem", data=seqsinDF, palette=[col2,col1])
sns.swarmplot(x="state", y="rem", data=seqsinDF, color='k', size=3)
plt.ylabel('Mean REMpre')
sns.despine()

fig4.savefig(outpath+'seqsin_rpre.pdf')

mean_ss_rem = [np.mean(seq_remmeans), np.mean(sin_remmeans)]
std_ss_rem = [np.std(seq_remmeans), np.std(sin_remmeans)]


