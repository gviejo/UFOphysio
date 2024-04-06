# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-03-30 19:11:51

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec
from itertools import combinations
# from functions import *
# import pynacollada as pyna
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

datasets = {'lmn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
            'adn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),            
            'psb':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),            
            }

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}

mean_spect = {g:{} for g in datasets.keys()}


for g in datasets.keys():

    for s in datasets[g]:

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
            windowsize = 0.05 # second
            N = int(windowsize/dt)*2
            count = 0.0

            #############################
            st = np.searchsorted(timestep, ufo_ts.t)
            st = st[st>N//2]
            st = st[st<len(timestep)-N//2-1]

            ent = []
                
            pwr = np.zeros((len(freq),N))
            
            # lfp = fp[:,sign_channels]

            batch_start = np.linspace(0, len(timestep), 200, dtype="int")

            for j, b in enumerate(batch_start[0:-1]):

                lfp = nap.TsdFrame(
                    t = timestep[b:batch_start[j+1]],
                    d = fp[b:batch_start[j+1],sign_channels],
                    columns = sign_channels
                    )

                dtype = np.complex128

                wavelet = signal.morlet2

                output = np.empty((lfp.shape[0], len(widths)), dtype=np.float64)
                for ind, width in enumerate(widths):
                    print(ind)                  
                    Nmin = np.min([10 * width, len(lfp)])
                    wavelet_data = np.conj(wavelet(Nmin, width, w)[::-1])                    
                    output[:,ind] = np.mean(lfp.convolve(wavelet_data.real), 1).d
                
                # cwtm = signal.cwt(lfp[b:batch_start[j+1]], signal.morlet2, widths, w=w)

                # sys.exit()

                idx = np.searchsorted(st, batch_start[j:j+2])

                for i, t in enumerate(st[idx[0]:idx[1]]):

                    pwr = output[t-N//2:t+N//2]                    
                    freqpwr = pwr.sum(0)
                    freqpwr = freqpwr/freqpwr.sum()

                    ent.append(np.sum(freqpwr*np.log2(freqpwr)))


            sys.exit()