# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-24 14:20:55

#%%

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
import matplotlib.pyplot as plt
from scipy.signal import stft


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

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


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
    channels = data.group_to_channel
    sign_channels = channels[ufo_channels[s][0]]
    ctrl_channels = channels[ufo_channels[s][1]]
    filename = data.basename + ".dat"    
            
    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)
    
    lfp = nap.TsdFrame(t=timestep, d=fp)

    ############################################################################################### 
    # SPECTROGRAM
    ###############################################################################################


    

    # for i in ufo_ep.index:
    #     tmp = timestep.restrict(ufo_ep.loc[[i]])
    #     lfp = fp[tmp.values[0]-10000:tmp.values[-1]+10000]
    #     lfp = lfp[:,channels]

    #     sys.exit()

ts_ex = [11005.398, 11005.982]

ep = nap.IntervalSet(start = ts_ex[1]-1, end = ts_ex[1]+1)

def compute_stft(s):
    Zx = []
    for i in range(s.shape[1]):
        # s = S[:,i]
        # s = s*np.blackman(s.size)

        f, t, Z = stft(s[:,i], fs=20000, window="blackman", nperseg=256, noverlap=200)

        # Zx.append(np.abs(Z))
        Zx.append(20*np.log10(np.abs(Z)))

    Zx = np.array(Zx)
    Zxx = Zx.mean(0)

    # freqs=np.geomspace(100, 1100, 10)
    # idx = np.digitize(f, freqs)
    # Zxx = np.array([Zxx[idx==i+1].mean(0) for i in range(freqs.shape[0])])

    t = t + s.t[0]
    return Zxx, t, f

def compute_multitaper(s):
    from mne.time_frequency import tfr_array_multitaper
    freqs=np.geomspace(100, 1200, 200)
    # freqs = np.arange(100, 1000, 200)
    Z = tfr_array_multitaper(
        s.d.T[None,:,:], 
        sfreq=20000, 
        freqs=freqs, 
        output = 'avg_power')
    
    return Z.mean(0), s.t, freqs

lfp2 = lfp.restrict(ep)

# Zx, tx, f = compute_multitaper(lfp2[:,sign_channels])
# Cx, tx, f = compute_multitaper(lfp2[:,ctrl_channels])

Zx, tx, f = compute_stft(lfp2[:,sign_channels])
Cx, tx, f = compute_stft(lfp2[:,ctrl_channels])

# t = np.arange(0, len(lfp2))*(1/20000)

Ax = Zx# - Cx



Ax = nap.TsdFrame(t=tx, d=Ax.T, columns = f.astype("int").astype("str"))

Ax.save(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/"+s.split("/")[-1]+"_TDF_Ex"))

# axs[0].plot(t, s.d)
# axs[0].set_xticklabels([])
# axs[0].set_xlim()
# # axs[0].axvline(0.125, color='red', alpha=0.5, lw=1)

# *_, im = axs[1].specgram(s, Fs=20000, NFFT=256, noverlap=200, cmap='jet', mode='magnitude')

figure()
subplot(221)
imshow(Zx, origin='lower', aspect='auto')
yticks(np.arange(Zx.shape[0])[::4], f[::4])
subplot(222)
imshow(Cx, origin='lower', aspect='auto')
yticks(np.arange(Cx.shape[0])[::4], f[::4])
subplot(212)
imshow(Ax.get(ts_ex[1]-0.2, ts_ex[1]+0.2).d.T, origin='lower', aspect='auto')
yticks(np.arange(Cx.shape[0])[::4], f[::4])
show()


# %%
