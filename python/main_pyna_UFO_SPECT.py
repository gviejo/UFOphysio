# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-18 16:35:13

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
            'ca1':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#'),            
            }

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}

mean_spect = {g:{} for g in datasets.keys()}


for g in datasets.keys():
# for g in ['ca1']:

    for s in datasets[g][0:1]:

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
            freq = np.linspace(100, 2000, 100)
            widths = w*fs / (2*freq*np.pi)
            windowsize = 0.05
            N = int(windowsize/dt)            
            pwr = np.zeros((len(freq),N))

            batch_size = 2**21
            start_batch = np.arange(0, len(fp), batch_size)
            count = 0

            for i, c in enumerate(sign_channels):

                for j, st in enumerate(start_batch[0:5]):
                    cwtm = signal.cwt(fp[st:st+batch_size,c], signal.morlet2, widths, w=w)

                    ts = ufo_ts.t[np.searchsorted(ufo_ts.t, timestep[st+N]):np.searchsorted(ufo_ts.t, timestep[np.minimum(st+batch_size-N,len(timestep)-N)])]
                    for k,t in enumerate(ts):
                        print(i/len(sign_channels),j/len(start_batch),k/len(ts), end="\r", flush=True)
                        idx = np.searchsorted(timestep[st:st+batch_size], t)
                        
                        tmp = np.abs(cwtm[:,idx-N//2:idx+N//2])
                        tmp /= tmp.sum(1)[:,np.newaxis]
                        pwr += tmp
                        count += 1
                
            pwr = pwr/count

            logfreq = np.geomspace(freq.min(), freq.max(), 30)
            freq_idx = np.digitize(freq, logfreq)-1
            logpwr = np.zeros((len(logfreq),pwr.shape[1]))

            for k in range(len(logfreq)):
                logpwr[k] = pwr[freq_idx==k].mean(0)


            mean_spect[g][s.split("/")[-1]] = logpwr

t = np.arange(0, N)*dt - (N//2)*dt

spect = {g:np.array([mean_spect[g][s] for s in mean_spect[g].keys()]).mean(0) for g in mean_spect}


figure(figsize=(20,4))
for i, g in enumerate(spect.keys()):
    subplot(1,4,i+1)
    imshow(spect[g], aspect='auto', origin='lower')
    title(g)
    yticks(np.arange(0, len(logfreq), 3), logfreq.astype("int")[::3])
    xticks(np.arange(0, N, 200), (t[np.arange(0, N, 200)]*1000).astype("int"))
    xlabel("Time (ms)")
    ylabel("Frequency")
show()