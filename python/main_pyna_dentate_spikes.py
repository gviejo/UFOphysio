# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-10-23 11:58:20
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-10-23 13:42:53


import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os

from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
from ufo_detection import *

from matplotlib.pyplot import *

# nap.nap_config.set_backend("jax")

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

datasets = ['ADN-HPC/B3214/B3218-241018']

for s in datasets:
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sleep_ep = data.epochs['sleep']
    # sws_ep = data.read_neuroscope_intervals('sws')
    #rem_ep = data.read_neuroscope_intervals('rem')
    try:
        # ufo_ep = nap.load_file(os.path.join(path, data.basename + '_ufo_ep.npz'))
        ufo_ep, ufo_ts = loadUFOs(path)
    except:
        pass
    
    ufo_ts = ufo_ts.restrict(sleep_ep)

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    filename = data.basename + ".eeg"

    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels, 1250)


    lfp = nap.TsdFrame(t=timestep, d=fp)

    mean_lfp = []
    ch = list(channels[0][32:])
    ch.remove(42)
    for c in ch:
        tmp1 = lfp[:,c]
        # tmp1 = tmp1 - np.mean(tmp1)
        # tmp1 = tmp1 / np.std(tmp1)
        tmp = nap.compute_perievent_continuous(tmp1, ufo_ts, minmax=(-1, 1))
        mean_lfp.append(np.mean(tmp, 1).d)

    mean_lfp = np.array(mean_lfp)


    # perilfp = nap.compute_perievent_continuous(lfp, ufo_ts, minmax=(-1, 1))
    T = tmp.t
    x = np.arange(0, mean_lfp.shape[1], 2000)
    figure(figsize=(16, 8))
    subplot(211)
    for i in range(len(mean_lfp)):
        plot(T, mean_lfp[i]-i*100)
    axvline(0)
    subplot(212)
    imshow(mean_lfp, aspect='auto')
    xticks(x, T[x])
    axvline(len(T)//2)
    show()