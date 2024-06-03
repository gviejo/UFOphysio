# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-24 16:46:10

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
from ufo_detection import *
from scipy.signal import butter, lfilter, filtfilt, hilbert

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_power(signal):
    frequency = 20000
    freq_band = (500, 1000)
    wsize = 41
    
    nSS = np.zeros(len(signal))
    
    for i in range(signal.shape[1]):
        print(i/signal.shape[1], flush=True)
        flfp = _butter_bandpass_filter(signal[:,0].d, 500, 1000, 20000)
        power = np.abs(hilbert(flfp))
        window = np.ones(wsize)/wsize            
        power = filtfilt(window, 1, power)
        nSS += power
        
    return nap.Tsd(t=signal.t, d=nSS)


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

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}

powers = {
    "ufos":{},
    "ctrl":{}
}

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    #sws_ep = data.read_neuroscope_intervals('sws')    
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]

    
    ufo_ep, ufo_ts = loadUFOs(path)

    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    # tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
    ###############################################################################################
    # MEMORY MAP
    ###############################################################################################
    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    sign_channels = channels[ufo_channels[s][0]]
    ctrl_channels = channels[ufo_channels[s][1]]
    filename = data.basename + ".dat"    

    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

    lfp = nap.TsdFrame(t=timestep, d=fp)

    for ch, name in zip([sign_channels, ctrl_channels], ['ufos', 'ctrl']):
        nSS = compute_power(lfp[:,ch])
        pwr = nSS.restrict(ufo_ep)
        powers[name][s.split("/")[-1]] = pwr.as_series()
    
    
import _pickle as cPickle
cPickle.dump(powers, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/mb_control.pickle"), 'wb'))

