# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-18 16:25:43

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
from scipy import signal
from scipy.ndimage import gaussian_filter

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

# for s in datasets:
for s in ['LMN-ADN/A5044/A5044-240402A']:
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

    lfp = nap.TsdFrame(t=timestep, d=fp)

    ############################################################################################### 
    # MORLET WAVELET SPECTROGRAM
    ###############################################################################################
    ep = nap.IntervalSet(11005.982-0.5, 11005.982+0.5)

    lfp = lfp.restrict(ep)[:,channels]

    fs = 20000.0
    dt = 1/fs
    w = 3.
    # freq = np.linspace(100, 2000, 100)
    freq = np.geomspace(100, 4000, 1000)
    widths = w*fs / (2*freq*np.pi)
    
    tfd = np.zeros((len(lfp), len(channels), len(freq)))

    for i in range(len(widths)):
        wavelet = signal.morlet2(10*widths[i], widths[i], w)

        wavelet = wavelet/np.sum(np.abs(wavelet))
        
        a = lfp.convolve(wavelet.real).d
        b = lfp.convolve(wavelet.imag).d
        tmp = np.abs(a+1j*b)
        tfd[:,:,i] = tmp

    tfd = np.mean(tfd, 1)
    tfd = nap.TsdFrame(t=lfp.t, d=tfd, columns = freq)

t = 11005.982
ex_ep = nap.IntervalSet(t-0.02, t+0.04)

tfd.save(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/"+s.split("/")[-1]+"_TFD_Ex.npz"))

figure()
subplot(211)
plot(lfp[:,0].restrict(ex_ep))
xlim(ex_ep[0,0], ex_ep[0,1])

subplot(212)
tmp = tfd.restrict(ex_ep).d
# tmp = tmp - np.mean(tmp, 0)
# tmp = tmp / np.std(tmp, 0)
imshow(gaussian_filter(tmp, 1).T , aspect='auto', origin='lower', extent = (ex_ep[0,0], ex_ep[0,1], freq[0], freq[-1]))
show()