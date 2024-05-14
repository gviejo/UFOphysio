# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-12 17:21:38

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *

############################################################################################### 
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

datasets = {
    'adn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    'lmn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    'psb':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')
}

ccs_long = {r:{e:[] for e in ['wak', 'rem', 'sws']} for r in ['adn', 'lmn', 'psb']}
ccs_short = {r:{e:[] for e in ['wak', 'rem', 'sws']} for r in ['adn', 'lmn', 'psb']}

SI_thr = {
    'adn':0.2, 
    'lmn':0.1,
    'psb':0.0
    }

for r in datasets.keys():
    for s in datasets[r]:

        print(s)

        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, s)
        data = nap.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        ufo_ep, ufo_ts = loadUFOs(path)

        idx = spikes._metadata[spikes._metadata["location"].str.contains(r)].index.values
        spikes = spikes[idx]

        if ufo_ts is not None:

            tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

            SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position['ry'].time_support.loc[[0]], minmax=(0,2*np.pi))

            spikes = spikes[SI[SI['SI']>SI_thr[r]].index.values]

            names = [s.split("/")[-1] + "_" + str(n) for n in spikes.keys()]

            for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):            
                cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.01, 0.6, ep, norm=True)
                cc.columns = names
                ccs_long[r][e].append(cc)

                cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.001, 0.015, ep, norm=True)
                cc.columns = names
                ccs_short[r][e].append(cc)


        else:
            print("No ufo in "+s)

for r in ccs_long.keys():
    for e in ccs_long[r].keys():
        ccs_long[r][e] = pd.concat(ccs_long[r][e], 1)
        ccs_short[r][e] = pd.concat(ccs_short[r][e], 1)

rcParams.update({'font.size': 15})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
lw = 2

figure(figsize = (10, 12))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

gs = GridSpec(4,3)
for i, r in enumerate(['lmn', 'adn', 'psb']):
    for j, e in enumerate(ccs_long[r].keys()):
        subplot(gs[i,j])
        if i == 0:
            title(e)
        if j == 0:
            ylabel(r, rotation=0, labelpad = 30)
        tmp = ccs_long[r][e].values
        tmp = tmp - tmp.mean(0)
        tmp = tmp / tmp.std(0)
        imshow(tmp.T, aspect='auto', cmap = 'jet')
        x = ccs_long[r][e].index.values
        xticks([0, len(x)//2, len(x)], [x[0], 0.0, x[-1]])

for j, e in enumerate(ccs_long[r].keys()):
    subplot(gs[-1,j])
    for i, r in enumerate(['lmn', 'adn', 'psb']):
        plot(ccs_long[r][e].mean(1), color = colors[i], linewidth=lw, label=r)
    axvline(0.0)
    xlim(x[0], x[-1])
    legend()
tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/ALL_CC_UFO_Long.png"))


figure(figsize = (10, 12))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

gs = GridSpec(4,3)
for i, r in enumerate(['lmn', 'adn', 'psb']):
    for j, e in enumerate(ccs_short[r].keys()):
        subplot(gs[i,j])
        if i == 0:
            title(e)
        if j == 0:
            ylabel(r, rotation=0, labelpad = 30)
        tmp = ccs_short[r][e].values
        tmp = tmp - tmp.mean(0)
        tmp = tmp / tmp.std(0)        
        tmp = tmp[:,np.where(~np.isnan(np.sum(tmp, 0)))[0]]
        imshow(tmp.T, aspect='auto', cmap = 'jet')
        x = ccs_short[r][e].index.values
        xticks([0, len(x)//2, len(x)], [x[0], 0.0, x[-1]])

for j, e in enumerate(ccs_short[r].keys()):
    subplot(gs[-1,j])
    for i, r in enumerate(['lmn', 'adn', 'psb']):
        plot(ccs_short[r][e].mean(1), color = colors[i], linewidth=lw, label=r)
    axvline(0.0)
    xlim(x[0], x[-1])
    legend()

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/ALL_CC_UFO_Short.png"))
