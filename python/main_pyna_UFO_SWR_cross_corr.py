# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-18 17:59:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-06-09 18:31:52
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
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

cc_short = {}
cc_long = {}

for s in datasets:
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
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)
    rip_ep, rip_ts = loadRipples(path)

    if ufo_ts is not None:        
        grp = nap.TsGroup({0:ufo_ts,1:rip_ts,}, evt = np.array(['ufo', 'rip']))

        ufo_cc = nap.compute_crosscorrelogram(grp, 0.5, 100, sws_ep)
        cc_long[s] = ufo_cc[(0,1)]

        ufo_cc = nap.compute_crosscorrelogram(grp, 0.01, 2, sws_ep)
        cc_short[s] = ufo_cc[(0,1)]


cc_long = pd.DataFrame.from_dict(cc_long)
cc_short = pd.DataFrame.from_dict(cc_short)

cc_long = cc_long.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
cc_short = cc_short.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)


figure()
subplot(121)
plot(cc_long, alpha = 0.5, linewidth=1)
plot(cc_long.mean(1), linewidth = 4, color = 'red')
xlabel("ufo/swr (s)")
subplot(122)
plot(cc_short, alpha = 0.5, linewidth=1)
plot(cc_short.mean(1), linewidth = 4, color = 'red')
xlabel("ufo/swr (s)")

show()