# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:50:26
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-06-14 17:38:03
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *

############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataGuillaume/'
# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')
# infos = getAllInfos(data_directory, datasets)

data_directory = '/mnt/Data2/'
datasets = ['LMN-PSB-2/A3018/A3018-220613A']
infos = getAllInfos(data_directory, datasets)

si = []
cc = []

for s in datasets:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)

    idx = spikes._metadata[spikes._metadata["location"].str.contains("psb")].index.values
    spikes = spikes[idx]
    spikes = spikes.getby_threshold('freq', 1, op = '>')

    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

    info = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position['ry'].time_support.loc[[0]], minmax=(0, 2*np.pi))['SI']

    #################################################################################################
    # CC 
    #################################################################################################

    ufo_cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.05, 2, sws_ep)

    names = [os.path.basename(s)+'_'+str(i) for i in ufo_cc.columns]

    info.index = names
    ufo_cc.columns = names

    si.append(info)
    cc.append(ufo_cc)

cc = pd.concat(cc, 1)
cc = cc.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2)
si = pd.concat(si, 0)

order = si.sort_values().index


figure()
subplot(121)
plot(cc, alpha =0.4, color = 'grey')
plot(cc.mean(1), linewidth=3)
subplot(122)
imshow(cc[order].values.T)


figure()
count = 1
neurons = order
for i,n in enumerate(neurons):
    subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,count, projection = 'polar')
    plot(tuning_curves[int(n.split('_')[1])], label = si.loc[n])
    count+=1
    gca().set_xticklabels([])

show()