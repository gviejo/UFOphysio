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

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions import *
from ufo_detection import *

from matplotlib.pyplot import *

# nap.nap_config.set_backend("jax")

from functions.functions import loadXML


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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN_DG.list'), delimiter = '\n', dtype = str, comments = '#')


perilfp = {}

for s in datasets:
# for s in ["ADN-HPC/B5100/B5102/B5102-250915"]:
    print(s)

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

    ufo_ep, ufo_ts = loadUFOs(path)


    if ufo_ep is None:
        print("No UFOs detected in this session {}".format(s))
    else:
        ufo_ts = ufo_ts.restrict(sleep_ep)

        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel
        filename = data.basename + ".eeg"

        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels, 1250)


        lfp = nap.TsdFrame(t=timestep, d=fp)

        # load XML info
        num_channels, fs, shank_to_channel, shank_to_keep = loadXML(path)

        mean_lfp = []
        ch = channels[0][shank_to_keep[0]]

        for c in ch:
            tmp1 = lfp[:,c]
            # tmp1 = tmp1 - np.mean(tmp1)
            # tmp1 = tmp1 / np.std(tmp1)
            tmp = nap.compute_perievent_continuous(tmp1, ufo_ts, minmax=(-1, 1))
            mean_lfp.append(np.nanmean(tmp, 1).d)

        mean_lfp = np.array(mean_lfp)
        mean_lfp = pd.DataFrame(mean_lfp.T, index=tmp.t, columns = ch)

        # perilfp = nap.compute_perievent_continuous(lfp, ufo_ts, minmax=(-1, 1))

        perilfp[s] = mean_lfp

###############################################################################################
# PLOTTING
###############################################################################################

# T = tmp.t
# x = np.arange(0, mean_lfp.shape[1], 2000)

figure(figsize=(16, 8))
gs = GridSpec(3, 4, hspace=0.4, wspace=0.4)
for i, s in enumerate(perilfp.keys()):
    mean_lfp = perilfp[s]
    gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i//3, i%3], height_ratios=[0.5, 0.5], hspace=0.1)
    subplot(gs2[0,0])
    for j in range(mean_lfp.shape[1]):
        plot(mean_lfp.index, mean_lfp.values[:,j]-j*100)
    axvline(0)
    title(s.split("/")[-1])
    subplot(gs2[1,0])
    imshow(mean_lfp.values.T, aspect='auto', extent=[mean_lfp.index[0], mean_lfp.index[-1], 0, mean_lfp.shape[1]])
    # xticks(x, T[x])
    # axvline(len(T)//2)

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_mean_HPC_LFP.pdf"))

