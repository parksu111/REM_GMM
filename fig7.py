#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:47:11 2021

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


#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path for figures
outpath = r'/home/cwlab08/Desktop/REM_GMM/final_figures/7_LightDark/'

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


#Read in gmm parameters for dark phase

darkgmmDF = pd.read_csv('darkgmmDF.csv')
darklinfitDF = pd.read_csv('darklinfitDF.csv')
darklogfitDF = pd.read_csv('darklogfitDF.csv')

dkhigh = darkgmmDF.khigh
dmlow = darkgmmDF.mlow
dmhigh = darkgmmDF.mhigh
dslow = darkgmmDF.slow
dshigh = darkgmmDF.shigh

dkhigh_loga = darklogfitDF.khigh[0]
dkhigh_logb = darklogfitDF.khigh[1]
dkhigh_logc = darklogfitDF.khigh[2]
dmlow_loga = darklogfitDF.mlow[0]
dmlow_logb = darklogfitDF.mlow[1]
dmlow_logc = darklogfitDF.mlow[2]
dmhigh_loga = darklogfitDF.mhigh[0]
dmhigh_logb = darklogfitDF.mhigh[1]
dmhigh_logc = darklogfitDF.mhigh[2]
dshigh_loga = darklogfitDF.shigh[0]
dshigh_logb = darklogfitDF.shigh[1]
dshigh_logc = darklogfitDF.shigh[2]

dslow_lina = darklinfitDF.slow[0]
dslow_linb = darklinfitDF.slow[1]

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


# Find intersection between low and high gaussian distributions (dark phase)
dintersection = []

for rem in np.arange(12.5,240,2.5):
    k1 = dkhigh_loga*np.log(rem+dkhigh_logb)+dkhigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = dmhigh_loga*np.log(rem+dmhigh_logb)+dmhigh_logc
    m2 = dmlow_loga*np.log(rem+dmlow_logb)+dmlow_logc
    s1 = dshigh_loga*np.log(rem+dshigh_logb)+dshigh_logc
    s2 = dslow_lina*rem+dslow_linb
    if k2>0:
        tfun = lambda x: k1*stats.norm.pdf(x,m1,s1)-k2*stats.norm.pdf(x,m2,s2)
        dintersection.append((rem, fsolve(tfun, x0=[5.25], maxfev=999999)))
        
dintersect_x = np.array([x[0] for x in dintersection])
dintersect_y = np.array([x[1] for x in dintersection])
dexp_intersect_y = np.exp(dintersect_y)



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


#subset dark phase dataframe
darkDF = darkDF.loc[darkDF.rem<240]
dnonDF = darkDF.loc[darkDF.rem<12.5]
dsubDF = darkDF.loc[darkDF.rem>=12.5]
dsubDF = dsubDF.reset_index(drop=True)

#Divide dark data into sequential and single cycles
dseqrows = []
dsinrows = []

for index,row in dsubDF.iterrows():
    rem = row['rem']
    sws = row['sws']
    if rem > max(dintersect_x):
        dsinrows.append(row)
    else:
        indBurst = np.where(dintersect_x==rem)[0][0]
        burstLim = dexp_intersect_y[indBurst][0]
        if sws<burstLim:
            dseqrows.append(row)
        else:
            dsinrows.append(row)

dseqDF = pd.DataFrame(dseqrows)
dsinDF = pd.DataFrame(dsinrows)


###############################################################################
#### Fig 7A, B
####Make dataframe containing percentage of each state for light and dark

#Compile light phase percentages
l_mname = []
l_wake = []
l_nrem = []
l_rem = []

for rec in recordings:
    print(rec)
    recs,nswitch,start = rp.find_light(ppath, rec)
    for idx,x in enumerate(recs):
        Mvec = rp.nts(x)
        Mtup = []
        if idx==0:
            Mtup = rp.vecToTup(Mvec, start=0)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
        fix_tup = rp.ma_thr(Mtup, threshold=8)
        rem = 0
        sws = 0
        wake = 0
        for y in fix_tup:
            if y[0]=='R':
                rem+=y[1]
            elif (y[0]=='N')or(y[0]=='MA'):
                sws+=y[1]
            else:
                wake+=y[1]
        l_mname.append(rec.split('_')[0])
        l_wake.append(wake)
        l_nrem.append(sws)
        l_rem.append(rem)
    
#Compile dark phase percentges
d_mname = []
d_wake = []
d_nrem = []
d_rem = []

for rec in recordings:
    print(rec)
    recs,nswitch,start = rp.find_light(ppath, rec, dark=True)
    for idx,x in enumerate(recs):
        Mvec = rp.nts(x)
        Mtup = []
        if idx==0:
            Mtup = rp.vecToTup2(Mvec, start=0)
        else:
            Mtup = rp.vecToTup(Mvec, start=start)
        fix_tup = rp.ma_thr(Mtup, threshold=8)
        rem = 0
        sws = 0
        wake = 0
        for y in fix_tup:
            if y[0]=='R':
                rem+=y[1]
            elif (y[0]=='N')or(y[0]=='MA'):
                sws+=y[1]
            else:
                wake+=y[1]
        d_mname.append(rec.split('_')[0])
        d_wake.append(wake)
        d_nrem.append(sws)
        d_rem.append(rem)

#Form dataframes
l_percDF = pd.DataFrame(list(zip(l_mname,l_wake,l_nrem,l_rem)),columns=['mname','wake','nrem','rem'])
d_percDF = pd.DataFrame(list(zip(d_mname,d_wake,d_nrem,d_rem)),columns=['mname','wake','nrem','rem'])

#list of mice
lmouse = list(set(l_percDF.mname))
dmouse = list(set(d_percDF.mname))

###Compile quantities by mice

#Light phase
lremp = []
lswsp = []
lwakep = []
lremos = []
lswsos = []
lphase = []

for mouse in lmouse:
    subDF = l_percDF.loc[l_percDF.mname==mouse]
    rem = sum(subDF.rem)
    sws = sum(subDF.nrem)
    wake = sum(subDF.wake)
    totp = rem+sws+wake
    totsleep = rem+sws
    lremp.append(rem/totp)
    lswsp.append(sws/totp)
    lwakep.append(wake/totp)
    lremos.append(rem/totsleep)
    lswsos.append(sws/totsleep)
    lphase.append('light')

#Dark phase
dremp = []
dswsp = []
dwakep = []
dremos = []
dswsos = []
dphase = []

for mouse in dmouse:
    subDF = d_percDF.loc[d_percDF.mname==mouse]
    rem = sum(subDF.rem)
    sws = sum(subDF.nrem)
    wake = sum(subDF.wake)
    totp = rem+sws+wake
    totsleep = rem+sws
    if totsleep>0:
        dremp.append(rem/totp)
        dswsp.append(sws/totp)
        dwakep.append(wake/totp)
        dremos.append(rem/totsleep)
        dswsos.append(sws/totsleep)
        dphase.append('dark')

#Form data frame for light and dark
lpercDF = pd.DataFrame(list(zip(lmouse,lremp,lswsp,lwakep,lremos,lswsos,lphase)),columns=['mname','remp','swsp','wakep','remos','swsos','phase'])
dpercDF = pd.DataFrame(list(zip(dmouse,dremp,dswsp,dwakep,dremos,dswsos,dphase)),columns=['mname','remp','swsp','wakep','remos','swsos','phase'])
totpercDF = pd.concat([lpercDF,dpercDF])


## Plot values
fig7A1 = plt.figure(figsize=(3,6))
plt.title('rem')
sns.swarmplot(x="phase",y="remp",data=totpercDF, color='k', size=3)
sns.barplot(x="phase",y="remp",data=totpercDF, palette=tpal)
sns.despine()

fig7A2 = plt.figure(figsize=(3,6))
plt.title('nrem')
sns.swarmplot(x="phase",y="swsp",data=totpercDF, color='k', size=3)
sns.barplot(x="phase",y="swsp",data=totpercDF, palette=tpal)
plt.ylim(0,0.82)
sns.despine()

fig7A3 = plt.figure(figsize=(3,6))
plt.title('wake')
sns.swarmplot(x="phase",y="wakep",data=totpercDF, color='k', size=3)
sns.barplot(x="phase",y="wakep",data=totpercDF, palette=tpal)
plt.ylim(0,0.82)
sns.despine()

fig7B = plt.figure(figsize=(3,6))
plt.title('remofs')
sns.swarmplot(x="phase",y="remos",data=totpercDF, color='k', size=3)
sns.barplot(x="phase",y="remos",data=totpercDF, palette=tpal)
sns.despine()


fig7A1.savefig(outpath+'f7a1.pdf')
fig7A2.savefig(outpath+'f7a2.pdf')
fig7A3.savefig(outpath+'f7a3.pdf')
fig7B.savefig(outpath+'f7b.pdf')


###Stats

## % of total

#REM
stats.levene(lpercDF.remp, dpercDF.remp)
stats.ttest_ind(lpercDF.remp, dpercDF.remp, equal_var=False)
#NREM
stats.levene(lpercDF.swsp, dpercDF.swsp)
stats.ttest_ind(lpercDF.swsp, dpercDF.swsp, equal_var=True)
#Wake
stats.levene(lpercDF.wakep, dpercDF.wakep)
stats.ttest_ind(lpercDF.wakep, dpercDF.wakep, equal_var=True)

## % of sleep
#REM
stats.levene(lpercDF.remos, dpercDF.remos)
stats.ttest_ind(lpercDF.remos, dpercDF.remos, equal_var=True)


###############################################################################
#### Fig7 C - Pie chart of single & sequential cycles

#list of labels and colors
lbls = ['Sequential','Single']
cols = ['gray','darkslateblue']

#Plot
fig7C = plt.figure()
plt.pie([len(dseqDF),len(dsinDF)], labels=lbls, autopct='%1.1f%%', colors=cols)

fig7C.savefig(outpath+'f7c_darksinseqpie.pdf')


###############################################################################
#### Fig 7D - Compare light and dark gmm parameters

tfun = lambda x: khigh_loga*np.log(x+khigh_logb)+khigh_logc - 1
kint = fsolve(tfun, x0=[160])

tfun2 = lambda x: dkhigh_loga*np.log(x+dkhigh_logb)+dkhigh_logc - 1
kint2 = fsolve(tfun2, x0=[160])

xes = np.arange(15,240,30)
xplot = np.arange(0,240,0.1)
xplot1a = np.arange(0,kint,0.1)
xplot2a = np.arange(0,kint2,0.1)

fig7D1 = plt.figure(figsize=(5,5))
plt.scatter(xes, khigh, s=10, color='mediumslateblue')
plt.plot(xplot1a,khigh_loga*np.log(xplot1a+khigh_logb)+khigh_logc, color='mediumslateblue',lw=1)
plt.scatter(xes, dkhigh, s=10, color='darkslateblue')
plt.plot(xplot2a,dkhigh_loga*np.log(xplot2a+dkhigh_logb)+dkhigh_logc, color='darkslateblue',lw=1)
plt.axhline(y=1, color='mediumslateblue',lw=1)
plt.axhline(y=1, color='darkslateblue',lw=1)
plt.xticks([0,100,200])
plt.xlim([0,240])
plt.ylim([0,1.1])
plt.yticks([0,0.5,1])
sns.despine()

fig7D2 = plt.figure(figsize=(5,5))
plt.scatter(xes, mhigh, s=10, color='mediumslateblue')
plt.plot(xplot,mhigh_loga*np.log(xplot+mhigh_logb)+mhigh_logc, color='mediumslateblue')
plt.scatter(xes, dmhigh, s=10, color='darkslateblue')
plt.plot(xplot,dmhigh_loga*np.log(xplot+dmhigh_logb)+dmhigh_logc, color='darkslateblue')
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()

fig7D3 = plt.figure(figsize=(5,5))
plt.scatter(xes, shigh, s=10, color='mediumslateblue')
plt.plot(xplot,shigh_loga*np.log(xplot+shigh_logb)+shigh_logc, color='mediumslateblue')
plt.scatter(xes, dshigh, s=10, color='darkslateblue')
plt.plot(xplot,dshigh_loga*np.log(xplot+dshigh_logb)+dshigh_logc, color='darkslateblue')
plt.xlim([0,240])
plt.xticks([0,100,200])
sns.despine()

fig7D4 = plt.figure(figsize=(5,5))
plt.scatter(xes, mlow, s=10, color='mediumslateblue')
plt.plot(xplot, mlow_loga*np.log(xplot+mlow_logb)+mlow_logc, color='mediumslateblue')
plt.scatter(xes, dmlow, s=10, color='darkslateblue')
plt.plot(xplot, dmlow_loga*np.log(xplot+dmlow_logb)+dmlow_logc, color='darkslateblue')
plt.xlim([0,240])
plt.ylim([3,6])
plt.xticks([0,100,200])
sns.despine()

fig7D5 = plt.figure(figsize=(5,5))
plt.scatter(xes, slow, s=10, color='mediumslateblue')
plt.plot(xplot, slow_lina*xplot+slow_linb, color='mediumslateblue')
plt.scatter(xes, dslow, s=10, color='darkslateblue')
plt.plot(xplot, dslow_lina*xplot+dslow_linb, color='darkslateblue')
plt.xlim([0,240])
plt.xticks([0,100,200])
plt.ylim([0,1])
plt.yticks([0,0.5,1])
sns.despine()


#save figures
fig7D1.savefig(outpath+'f7d_khigh.pdf')
fig7D2.savefig(outpath+'f7d_mhigh.pdf')
fig7D3.savefig(outpath+'f7d_shigh.pdf')
fig7D4.savefig(outpath+'f7d_mlow.pdf')
fig7D5.savefig(outpath+'f7d_slow.pdf')


###############################################################################
#### Fig 7E - Bootstrap results

lightDF = lightDF.reset_index(drop=True)
darkDF = darkDF.reset_index(drop=True)


## Perform 10,000 bootstraps on light and dark datasets and estimate gmm parameters
allKlows = []
allKhighs = []
allMhighs = []
allMlows = []
allShighs = []
allSlows = []
allProportions = []

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
        row = lightDF.iloc[x]
        rem = row['rem']
        logsws = row['logsws']
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
    allKhighs.append(khigh)
    allKlows.append(klow)
    allMhighs.append(mhigh)
    allMlows.append(mlow)
    allShighs.append(shigh)
    allSlows.append(slow)
    allProportions.append([len(logsws30),len(logsws60),len(logsws90),len(logsws120),len(logsws150),len(logsws180),len(logsws210),len(logsws240)])
    print('l-'+str(i))

d_allKlows = []
d_allKhighs = []
d_allMhighs = []
d_allMlows = []
d_allShighs = []
d_allSlows = []
d_all_Proportions = []

exception_cnt = 0
i = 0
while len(d_allKlows)<10000:
    np.random.seed(i)
    print('d'+str(i))
    sample = np.random.choice(1242,1242)
    logsws30 = []
    logsws60 = []
    logsws90 = []
    logsws120 = []
    logsws150 = []
    logsws180 = []
    logsws210 = []
    logsws240 = []
    for x in sample:
        row = darkDF.iloc[x]
        rem = row['rem']
        logsws = row['logsws']
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
    if (len(logsws210)>=2)&(len(logsws240)>=2):
        khigh,klow,mhigh,mlow,shigh,slow = rp.dark_gmm(logdatas)
        d_allKhighs.append(khigh)
        d_allKlows.append(klow)
        d_allMhighs.append(mhigh)
        d_allMlows.append(mlow)
        d_allShighs.append(shigh)
        d_allSlows.append(slow)
        d_all_Proportions.append([len(logsws30),len(logsws60),len(logsws90),len(logsws120),len(logsws150),len(logsws180),len(logsws210),len(logsws240)])        
    else:
        exception_cnt+=1
        i+=1
    i+=1


### For parameters estimated through bootsrap, fit functions relating parameters to REMpre

#Define log function
def log_func(x,a,b,c):
    return a*np.log(x+b)+c

#x values
xes = np.arange(15,240,30)

#Lists to store coefficients of function
lkhighas = []
lkhighbs = []
lkhighcs = []

dkhighas = []
dkhighbs = []
dkhighcs = []

for i in range(10000):
    print(i)
    khighs = allKhighs[i]
    khigh_log = curve_fit(log_func, xes[0:6], khighs[0:6], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]),maxfev=9999999)[0]
    khigh_loga = khigh_log[0]
    khigh_logb = khigh_log[1]
    khigh_logc = khigh_log[2]
    lkhighas.append(khigh_loga)
    lkhighbs.append(khigh_logb)
    lkhighcs.append(khigh_logc)
    dkhighs = d_allKhighs[i]
    dkhigh_log = curve_fit(log_func, xes[0:4], khighs[0:4], bounds=([-np.inf, 0, -np.inf],[np.inf,np.inf,np.inf]),maxfev=9999999)[0]
    dkhigh_loga = dkhigh_log[0]
    dkhigh_logb = dkhigh_log[1]
    dkhigh_logc = dkhigh_log[2]
    dkhighas.append(dkhigh_loga)
    dkhighbs.append(dkhigh_logb)
    dkhighcs.append(dkhigh_logc)


lmhighas = []
lmhighbs = []
lmhighcs = []

dmhighas = []
dmhighbs = []
dmhighcs = []

for i in range(10000):
    print(i)
    mhighs = allMhighs[i]
    mhigh_log = curve_fit(log_func, xes, mhighs, bounds=([-np.inf,0,-np.inf],[np.inf,np.inf,np.inf]))[0]
    mhigh_loga = mhigh_log[0]
    mhigh_logb = mhigh_log[1]
    mhigh_logc = mhigh_log[2]
    lmhighas.append(mhigh_loga)
    lmhighbs.append(mhigh_logb)
    lmhighcs.append(mhigh_logc)
    dmhighs = d_allMhighs[i]
    dmhigh_log = curve_fit(log_func, xes, dmhighs, bounds=([-np.inf,0,-np.inf],[np.inf,np.inf,np.inf]), maxfev=9999999)[0]
    dmhigh_loga = dmhigh_log[0]
    dmhigh_logb = dmhigh_log[1]
    dmhigh_logc = dmhigh_log[2]
    dmhighas.append(dmhigh_loga)
    dmhighbs.append(dmhigh_logb)
    dmhighcs.append(dmhigh_logc)    

lshighas = []
lshighbs = []
lshighcs = []

dshighas = []
dshighbs = []
dshighcs = []

for i in range(10000):
    print(i)
    shighs = allShighs[i]
    shigh_log = curve_fit(log_func, xes, shighs, p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
    shigh_loga = shigh_log[0]
    shigh_logb = shigh_log[1]
    shigh_logc = shigh_log[2]
    dshighs = d_allShighs[i]
    dshigh_log = curve_fit(log_func, xes, dshighs, p0=[-1,1,1],bounds=([-np.inf, 0, -np.inf],[np.inf, np.inf, np.inf]), maxfev=9999999)[0]
    dshigh_loga = dshigh_log[0]
    dshigh_logb = dshigh_log[1]
    dshigh_logc = dshigh_log[2]    
    lshighas.append(shigh_loga)
    lshighbs.append(shigh_logb)
    lshighcs.append(shigh_logc)
    dshighas.append(dshigh_loga)
    dshighbs.append(dshigh_logb)
    dshighcs.append(dshigh_logc)    

    
###### Stats
lxes = np.arange(7.5,240,2.5)
dxes = np.arange(12.5,240,2.5)

#khigh
lklongavgs = []
dklongavgs = []

for i in range(10000):
    lk_a = lkhighas[i]
    lk_b = lkhighbs[i]
    lk_c = lkhighcs[i]
    dk_a = dkhighas[i]
    dk_b = dkhighbs[i]
    dk_c = dkhighcs[i]
    lres = []
    dres = []
    lfun = lambda x: lk_a*np.log(x+lk_b)+lk_c
    dfun = lambda x: dk_a*np.log(x+dk_b)+dk_c
    for xval in lxes:
        res = lfun(xval)
        if res>1:
            lres.append(1)
        else:
            lres.append(res)
    for xval in dxes:
        res = dfun(xval)
        if res>1:
            dres.append(1)
        else:
            dres.append(res)
    lklongavgs.append(np.mean(lres))
    dklongavgs.append(np.mean(dres))

stats.levene(lklongavgs, dklongavgs)
stats.ttest_ind(lklongavgs, dklongavgs, equal_var=False)
        

#mhigh
dmlongavgs = []
lmlongavgs = []

for i in range(10000):
    lm_a = lmhighas[i]
    lm_b = lmhighbs[i]
    lm_c = lmhighcs[i]
    dm_a = dmhighas[i]
    dm_b = dmhighbs[i]
    dm_c = dmhighcs[i]
    lmfun = lambda x: lm_a*np.log(x+lm_b)+lm_c
    dmfun = lambda x: dm_a*np.log(x+dm_b)+dm_c
    lmlongavgs.append(np.mean(lmfun(lxes)))
    dmlongavgs.append(np.mean(dmfun(dxes)))

stats.levene(lmlongavgs, dmlongavgs)
stats.ttest_ind(lmlongavgs, dmlongavgs, equal_var=False)


#shigh
dslongavgs = []
lslongavgs = []

for i in range(10000):
    lm_a = lshighas[i]
    lm_b = lshighbs[i]
    lm_c = lshighcs[i]
    dm_a = dshighas[i]
    dm_b = dshighbs[i]
    dm_c = dshighcs[i]
    lmfun = lambda x: lm_a*np.log(x+lm_b)+lm_c
    dmfun = lambda x: dm_a*np.log(x+dm_b)+dm_c
    lslongavgs.append(np.mean(lmfun(lxes)))
    dslongavgs.append(np.mean(dmfun(dxes)))

stats.levene(lslongavgs, dslongavgs)
stats.ttest_ind(lslongavgs, dslongavgs, equal_var=False)

## Plot

#x values to plot
xplot = np.arange(12.5,241,1)

# Calculate mu_high(long) of light and dark bootstraps
darkmeanplot = []
lightmeanplot = []

for i in range(10000):
    a = lmhighas[i]
    b = lmhighbs[i]
    c = lmhighcs[i]
    a2 = dmhighas[i]
    b2 = dmhighbs[i]
    c2 = dmhighcs[i]
    darkmeanplot.append(np.exp(log_func(xplot,a2,b2,c2)))
    lightmeanplot.append(np.exp(log_func(xplot,a,b,c)))

## Calculate refractory duration of light and dark bootstraps
refxes = np.arange(12.5,242.5,2.5)
lightrefplot = []
darkrefplot = []

for i in range(10000):
    ma = lmhighas[i]
    mb = lmhighbs[i]
    mc = lmhighcs[i]
    ma2 = dmhighas[i]
    mb2 = dmhighbs[i]
    mc2 = dmhighcs[i]
    sa = lshighas[i]
    sb = lshighbs[i]
    sc = lshighcs[i]
    sa2 = dshighas[i]
    sb2 = dshighbs[i]
    sc2 = dshighcs[i]
    lref = []
    dref = []
    for rem in refxes:
        lmean = ma*np.log(rem+mb)+mc
        lstd = sa*np.log(rem+sb)+sc
        dmean = ma2*np.log(rem+mb2)+mc2
        dstd = sa2*np.log(rem+sb2)+sc2
        lref.append(lmean-2.326*lstd)
        dref.append(dmean-2.326*dstd)
    lightrefplot.append(np.exp(lref))
    darkrefplot.append(np.exp(dref))
    

# Confidence intervals for mu_high
alight = np.percentile(lightmeanplot, 2.5, axis=0)
blight = np.percentile(lightmeanplot, 97.5, axis=0)
adark = np.percentile(darkmeanplot, 2.5, axis=0)
bdark = np.percentile(darkmeanplot, 97.5, axis=0)

# Confidence intervals for refractory period
arlight = np.percentile(lightrefplot, 2.5, axis=0)
brlight = np.percentile(lightrefplot, 97.5, axis=0)
ardark = np.percentile(darkrefplot, 2.5, axis=0)
brdark = np.percentile(darkrefplot, 97.5, axis=0)

#Plot
fig7E = plt.figure()
plt.plot(xplot, np.mean(lightmeanplot,axis=0), color='mediumslateblue', lw=1)
plt.plot(xplot, np.mean(darkmeanplot,axis=0), color='darkslateblue', lw=1)
plt.fill_between(xplot, alight, blight, color='mediumslateblue', alpha=0.3)
plt.fill_between(xplot, adark, bdark, color='darkslateblue', alpha=0.3)
plt.plot(refxes, np.mean(lightrefplot,axis=0), color='mediumslateblue', lw=1, ls='--')
plt.plot(refxes, np.mean(darkrefplot,axis=0), color='darkslateblue', lw=1, ls='--')
plt.fill_between(refxes,arlight,brlight,color='mediumslateblue', alpha=0.3)
plt.fill_between(refxes,ardark,brdark,color='darkslateblue',alpha=0.3)
sns.despine()
plt.xlim([0,240])

fig7E.savefig(outpath+'f7e_lightdark_bootstrap.pdf')


############################################################################### 
#### Fig 7F - CDFs of light and dark phases for different values of REMpre

#relaod model parameters
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


#Read in gmm parameters for dark phase

darkgmmDF = pd.read_csv('darkgmmDF.csv')
darklinfitDF = pd.read_csv('darklinfitDF.csv')
darklogfitDF = pd.read_csv('darklogfitDF.csv')

dkhigh = darkgmmDF.khigh
dmlow = darkgmmDF.mlow
dmhigh = darkgmmDF.mhigh
dslow = darkgmmDF.slow
dshigh = darkgmmDF.shigh

dkhigh_loga = darklogfitDF.khigh[0]
dkhigh_logb = darklogfitDF.khigh[1]
dkhigh_logc = darklogfitDF.khigh[2]
dmlow_loga = darklogfitDF.mlow[0]
dmlow_logb = darklogfitDF.mlow[1]
dmlow_logc = darklogfitDF.mlow[2]
dmhigh_loga = darklogfitDF.mhigh[0]
dmhigh_logb = darklogfitDF.mhigh[1]
dmhigh_logc = darklogfitDF.mhigh[2]
dshigh_loga = darklogfitDF.shigh[0]
dshigh_logb = darklogfitDF.shigh[1]
dshigh_logc = darklogfitDF.shigh[2]

dslow_lina = darklinfitDF.slow[0]
dslow_linb = darklinfitDF.slow[1]

xplot = np.arange(0,2500,1)

fig7F = plt.figure(figsize=(12,3))
plt.subplots_adjust(hspace=0.4)
for idx,rem in enumerate(np.arange(50,250,50)):
    plt.subplot(1,4,idx+1)
    k1 = khigh_loga*np.log(rem+khigh_logb)+khigh_logc
    if k1>1:
        k1=1
    k2 = 1-k1
    m1 = mhigh_loga*np.log(rem+mhigh_logb)+mhigh_logc
    m2 = mlow_loga*np.log(rem+mlow_logb)+mlow_logc
    s1 = shigh_loga*np.log(rem+shigh_logb)+shigh_logc
    s2 = slow_lina*rem+slow_linb
    dk1 = dkhigh_loga*np.log(rem+dkhigh_logb)+dkhigh_logc
    if dk1>1:
        dk1=1
    dk2 = 1-dk1
    dm1 = dmhigh_loga*np.log(rem+dmhigh_logb)+dmhigh_logc
    dm2 = dmlow_loga*np.log(rem+dmlow_logb)+dmlow_logc
    ds1 = dshigh_loga*np.log(rem+dshigh_logb)+dshigh_logc
    ds2 = dslow_lina*rem+dslow_linb
    
    if k2>0:
        lfun = lambda x: k1*stats.lognorm.cdf(x,s1,0,np.exp(m1))+k2*stats.lognorm.cdf(x,s2,0,np.exp(m2))
    else:
        lfun = lambda x: k1*stats.lognorm.cdf(x,s1,0,np.exp(m1))
    if dk2>0:
        dfun = lambda x: dk1*stats.lognorm.cdf(x,ds1,0,np.exp(dm1))+dk2*stats.lognorm.cdf(x,ds2,0,np.exp(dm2))
    else:
        dfun = lambda x: dk1*stats.lognorm.cdf(x,ds1,0,np.exp(dm1))
    plt.title('REM='+str(rem))
    plt.plot(xplot, lfun(xplot), color='mediumslateblue')
    plt.plot(xplot, dfun(xplot), color='darkslateblue')
    sns.despine()
    plt.ylim([0,1])
    plt.yticks([0,0.5,1])

fig7F.savefig(outpath+'f7f_cdf.pdf')


