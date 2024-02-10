# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-02-07 10:12:31

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
from multiprocessing import Pool
import functools

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

datasets = {'lmn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
            'adn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),            
            'psb':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),            
            }

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}

mean_spect = {g:{} for g in datasets.keys()}


for g in datasets.keys():
# for g in ['ca1']:

    for s in datasets[g]:

        print(s)
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, s)
        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        ufo_ep, ufo_ts = loadUFOs(path)

        if ufo_ts is not None:        

            ###############################################################################################
            # MEMORY MAP
            ###############################################################################################
            data.load_neurosuite_xml(data.path)
            channels = data.group_to_channel

            if g == "lmn":
                sign_channels = channels[ufo_channels[s][0]]
            else:
                sign_channels = channels[np.unique(spikes.getby_category("location")[g].get_info("group"))[0]]

            filename = data.basename + ".dat"
                    
            fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)            

            fs = 20000.0
            dt = 1/fs
            w = 5.
            # freq = np.linspace(100, 2000, 100)
            freq = np.geomspace(100, 2000, 200)
            widths = w*fs / (2*freq*np.pi)
            windowsize = 0.05
            N = int(windowsize/dt)*2          
            pwr = np.zeros((len(freq),N))
            count = 0.0

            #############################
            st = np.searchsorted(timestep, ufo_ts.t)
            st = st[st>N//2]
            st = st[st<len(timestep)-N//2-1]

            def func(args):                
                channel, lfp, freq, N, sign_channels, windowsize, dt, widths, w = args
                pwr2 = np.zeros((len(freq),N))
                count2 = 0.0
                cwtm = signal.cwt(lfp, signal.morlet2, widths, w=w)
                tmp = np.abs(cwtm)
                tmp /= tmp.sum(1)[:,np.newaxis]
                return tmp

            n_core = len(sign_channels)            
            p = Pool(n_core)

            for i, t in enumerate(st):
          
                print(s, 100*i/len(ufo_ts))

                items = []
                lfp = fp[t-N//2:t+N//2,:]
                for j, c in enumerate(sign_channels):                    
                    items.append((c, np.array(lfp[:,c]), freq, N, sign_channels, windowsize, dt, widths, w))

                tmp = p.map_async(func, items).get()

                pwr += np.sum(np.array(tmp), 0)
                count += len(sign_channels)
                # sys.exit()
                # for j, c in enumerate(sign_channels):
                
                #     cwtm = signal.cwt(fp[st-int(windowsize/dt):st+int(windowsize/dt),c], signal.morlet2, widths, w=w)
                #     tmp = np.abs(cwtm)
                #     tmp /= tmp.sum(1)[:,np.newaxis]
                #     pwr += tmp
                #     count += 1.0
                
            pwr = pwr/count

            # sys.exit()

            # logfreq = np.geomspace(freq.min(), freq.max(), 30)
            # freq_idx = np.digitize(freq, logfreq)-1
            # logpwr = np.zeros((len(logfreq),pwr.shape[1]))

            # for k in range(len(logfreq)):
            #     logpwr[k] = pwr[freq_idx==k].mean(0)

            # sys.exit()

            mean_spect[g][s.split("/")[-1]] = pwr

t = np.arange(0, N)*dt - (N//2)*dt

spect = {g:np.array([mean_spect[g][s] for s in mean_spect[g].keys()]).mean(0) for g in mean_spect}


figure(figsize=(20,10))
gs = GridSpec(2,3)
for i, g in enumerate(spect.keys()):
    subplot(gs[0,i])
    imshow(spect[g], aspect='auto', origin='lower')
    title(g)
    # yticks(np.arange(0, len(logfreq), 3), logfreq.astype("int")[::3])
    yticks(np.arange(0, len(freq), 20), freq.astype("int")[::20])
    xticks(np.arange(0, N, 200), (t[np.arange(0, N, 200)]*1000).astype("int"))
    xlabel("Time (ms)")
    ylabel("Frequency")

    subplot(gs[1,i])
    semilogx(freq, spect[g][:,1000])        
    xlabel("Frequency")

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_SPECTROGRAM.png"))

