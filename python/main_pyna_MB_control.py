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
from scipy.signal import hilbert

# def _butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
#
# def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
#     b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
#
#
# def compute_power(signal: nap.TsdFrame) -> nap.Tsd:
#
#     wsize = 41
#
#     nSS = np.zeros(len(signal))
#
#     for i in range(signal.shape[1]):
#         print(i/signal.shape[1], flush=True)
#         flfp = _butter_bandpass_filter(signal[:,0].d, 500, 1000, 20000)
#         power = np.abs(hilbert(flfp))
#         window = np.ones(wsize)/wsize
#         power = filtfilt(window, 1, power)
#         nSS += power
#
#     return nap.Tsd(t=signal.t, d=nSS)


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
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.exists(os.path.join(path, "pynapplenwb")):

        print(s)

        data = ntm.load_session(path, 'neurosuite')

        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        #sws_ep = data.read_neuroscope_intervals('sws')
        sleep_ep = data.epochs['sleep']
        spikes = spikes[spikes.location=="lmn"]


        ufo_ep, ufo_ts = loadUFOs(path)

        ufo_ep = ufo_ep.intersect(sleep_ep[0])
        ufo_ts = ufo_ts.restrict(sleep_ep[0])

        if ufo_ts is None:
            print("No UFOs detected for this session {}".format(s))
        else:

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

            epochs = lfp.time_support.split(3*60) # 3 minutes chunks

            ###############################################################################################
            # COMPUTING POWER
            ###############################################################################################
            for chs, name in zip([sign_channels, ctrl_channels], ['ufos', 'ctrl']):
                nSS = np.zeros(len(lfp.restrict(ufo_ep)))

                for i, ch in enumerate(chs):
                    print("{} - Channel {}/{}".format(name, i+1, len(chs)), flush=True)
                    flfp = nap.apply_bandpass_filter(lfp[:, ch], (500, 2000), fs=20000, order=4)
                    power = np.abs(hilbert(flfp.d))
                    wsize = 41
                    window = np.ones(wsize) / wsize
                    power = filtfilt(window, 1, power)
                    power = power - np.mean(power)
                    power = power / np.std(power)
                    power = nap.Tsd(t=lfp.t, d=power)
                    power = power.restrict(ufo_ep)

                    nSS += power.values

                nSS = nSS / len(chs)
                # nSS = nSS - np.mean(nSS)
                # nSS = nSS / np.std(nSS)
                # nSS = nap.Tsd(t=lfp.t, d=nSS)
                # pwr = nSS.restrict(ufo_ep)
                # nSS = np.array(nSS)
                powers[name][s.split("/")[-1]] = nSS


    
import _pickle as cPickle
cPickle.dump(powers, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/mb_control.pickle"), 'wb'))

