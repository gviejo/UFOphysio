# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-01-18 16:37:20
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-18 19:41:57

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d

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

############################################################################################### 
# LMN - ADN
###############################################################################################
sessions = ["LMN-ADN/A5043/A5043-230306A", "LMN-PSB/A3018/A3018-220614B"]

times = [
    [1137.9, 1141.072, 1155.630, 1167.528],
    [7.633, 15.479, 35.901, 48.474, 4311.211]
]

names = ['adn', 'psb']

for i, s in enumerate(sessions):
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)
    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel

    filename = data.basename + ".dat"
            
    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

    for t in times[i]:

        st = np.searchsorted(timestep, t)

        fs = 20000.0
        dt = 1/fs
        w = 10.
        # freq = np.linspace(100, 5000, 1000)
        freq = np.geomspace(100, 10000, 1000)
        widths = w*fs / (2*freq*np.pi)
        windowsize = 0.05
        N = int(windowsize/dt)*2

        # logfreq = np.geomspace(freq.min(), freq.max(), 100)
        # freq_idx = np.digitize(freq, logfreq)-1

        pwrs = {g:np.zeros((len(freq),N)) for g in ['lmn', names[i]]}

        if names[i] == 'adn':
            adn_channels = channels[2]
            lmn_channels = channels[4]
        elif names[i] == 'psb':
            adn_channels = channels[0][len(channels[0])//2:]
            lmn_channels = channels[1]
        


        lfps = {}
        cwts = {}

        for g, chs in zip([names[i], 'lmn'], [adn_channels, lmn_channels]):

            pwrs[g] = []

            for c in chs:

                lfps[g] = fp[st-int(windowsize/dt):st+int(windowsize/dt),c]

                cwtm = signal.cwt(lfps[g], signal.morlet2, widths, w=w)

                cwts[g] = cwtm.real

                tmp = np.abs(cwtm)
                tmp /= tmp.sum(1)[:,np.newaxis]
            
            # logpwr = np.zeros((len(logfreq),N))
            # for k in range(len(logfreq)):
            #     logpwr[k] = tmp[freq_idx==k].mean(0)
            # pwrs[g] = logpwr

                pwrs[g].append(tmp)

            pwrs[g] = np.array(pwrs[g]).mean(0)



        figure(figsize=(20, 15))
        gs = GridSpec(2, 1)
        for k, g in enumerate([names[i], 'lmn']):
            gs2 = GridSpecFromSubplotSpec(2, 1, gs[k,0])

            subplot(gs2[0,0])
            pwr = gaussian_filter(pwrs[g], 40)
            imshow(pwr, aspect='auto', origin='lower', cmap='jet')
            title(g)
            yticks(np.arange(0, len(freq), 200), freq.astype("int")[::200])
            # xticks(np.arange(0, N, 200), (t[np.arange(0, N, 200)]*1000).astype("int"))
            max_pos = np.unravel_index(np.argmax(pwr), pwrs[g].shape)
            print(g, freq[max_pos[0]])    
            plot(max_pos[1], max_pos[0], 'o')


            subplot(gs2[1,0])
            xlabel("Time (ms)")
            ylabel("Frequency")

            subplot(gs2[1,0])
            plot(lfps[g]-gaussian_filter1d(lfps[g], 100), linewidth = 0.7)
            xlim(0, N)

            # subplot(gs2[2,0])
            plot(cwts[g][max_pos[0]], linewidth = 0.7)

            xlim(0, N)


        tight_layout()

        
        savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/examples/"+s.split("/")[-1]+"_"+str(t)+".png"))