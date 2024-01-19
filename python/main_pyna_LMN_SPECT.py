# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-12-02 17:38:44

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
# from functions import *
# import pynacollada as pyna
from ufo_detection import *
from matplotlib.pyplot import *

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

# for s in datasets:
for s in ['LMN-ADN/A5011/A5011-201014A']:
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    #rem_ep = data.read_neuroscope_intervals('rem')
    try:
        ufo_ep = nap.load_file(os.path.join(path, data.basename + '_ufo_ep.npz'))
    except:
        pass
    # ufo_ep, ufo_ts = loadUFOs(path)

    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel[np.unique(spikes._metadata["group"].values)[0]]    
    filename = data.basename + ".dat"    
            
    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

    timestep = nap.Tsd(t=timestep, d=np.arange(len(timestep)))

    ############################################################################################### 
    # SPECTROGRAM
    ###############################################################################################


    # for i in ufo_ep.index:
    #     tmp = timestep.restrict(ufo_ep.loc[[i]])
    #     lfp = fp[tmp.values[0]-10000:tmp.values[-1]+10000]
    #     lfp = lfp[:,channels]

    #     sys.exit()


ep = nap.IntervalSet(start = 14870.909, end = 14871.909)

tmp = timestep.restrict(ep)
lfp = fp[tmp.values[0]:tmp.values[-1]]
lfp = lfp[:,channels]


import matplotlib.pyplot as plt
s = lfp[:,0]
t = np.arange(0, len(s))*(1/20000)
s = s*np.blackman(s.size)


fig, axs = plt.subplots(figsize=(15, 5), nrows=2, gridspec_kw={'height_ratios':[2, 5]})

axs[0].plot(t, s)
axs[0].set_xticklabels([])
axs[0].set_xlim()
# axs[0].axvline(0.125, color='red', alpha=0.5, lw=1)

*_, im = axs[1].specgram(s, Fs=20000, NFFT=256, noverlap=200, cmap='jet', mode='magnitude')
axs[1].tick_params(axis='both', which='major', labelsize=16)
# cbar = plt.colorbar(im, aspect=10)
# cbar.ax.tick_params(labelsize=12) 
axs[1].grid(alpha=0.3)
#axs[1].grid(False)
axs[1].set_ylim(0, 2000)
axs[1].set_ylabel("↑ freq [Hz]")
axs[1].set_xlabel("time [s] →")

plt.tight_layout()
plt.show()

