#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:59:41 2021

@author: cwlab08
"""


#Import necessary packages
import os
import rempropensity as rp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

#Set path for recordings
ppath = r'/home/cwlab08/Desktop/24hours/'
recordings = os.listdir(ppath)

#Set path to output figures
outpath = r'/home/cwlab08/Desktop/final_figures/Sfig1/'

#Make dataframe containing all REM-NREM cycles
remDF = rp.standard_recToDF(ppath, recordings, 8, False)
remDF['loginter'] = remDF['inter'].apply(np.log)

###############################################################################

splitDFs = rp.splitData(remDF,30)

nremmeans = [np.mean(x.logsws) for x in splitDFs]
nremstds = [np.std(x.logsws) for x in splitDFs]

intermeans = [np.mean(x.loginter) for x in splitDFs]
interstds = [np.std(x.loginter) for x in splitDFs]

xes = np.arange(15,240,30)

sfig1A = plt.figure()
plt.plot(xes, nremmeans, color='k')
plt.errorbar(xes, nremmeans, nremstds, color='k')
sns.despine()
plt.xlabel('REMpre (s)')
plt.ylabel('ln(|N|)')
plt.xticks([0,60,120,180,240])
plt.yticks([4,5,6,7,8])

sfig1B = plt.figure()
plt.plot(xes, intermeans, color='k')
plt.errorbar(xes, intermeans, interstds, color='k')
sns.despine()
plt.xlabel('REMpre (s)')
plt.ylabel('ln(inter)')
plt.xticks([0,60,120,180,240])
plt.yticks([4,5,6,7,8])


sfig1A.savefig(outpath+'sf1_lognrem.pdf')
sfig1B.savefig(outpath+'sf1_loginter.pdf')