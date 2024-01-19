# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-14 19:05:19

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
from scipy.signal import hilbert

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
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#'),    
    ])

rip_ch = {'A5022':3,'A5026':4, 'A5027':4, 'A5030':11, 'A5030-220220A':1}

ufo_theta = {'rem':{}, 'wak':{}}

for s in datasets:

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
        ufo_gr = nap.TsGroup({0:ufo_ts})

        ############################################################################################### 
        # LOAD CA1 channel
        ############################################################################################### 
        lfp = data.load_lfp(channel=rip_ch[data.basename.split("-")[0]],extension='.eeg',frequency=1250.0)

        lfp2 = lfp.bin_average(5/lfp.rate)
        lfp_filt = pyna.eeg_processing.bandpass_filter(lfp2, 5, 15, int(lfp2.rate), 2)
        lfp_filt = nap.Tsd(t=lfp_filt.t, d=hilbert(lfp_filt))
        power = np.abs(lfp_filt)
        power = power.bin_average(0.01)
        power = power.smooth(1, 100)


        pwr_wak = nap.compute_event_trigger_average(ufo_gr, power, 0.05, (-5.0, 5.0), wake_ep)
        pwr_rem = nap.compute_event_trigger_average(ufo_gr, power, 0.05, (-5.0, 5.0), rem_ep)


        ufo_theta['rem'][s.split("/")[-1]] = pwr_rem[:,0].as_series()
        ufo_theta['wak'][s.split("/")[-1]] = pwr_wak[:,0].as_series()

for k in ufo_theta.keys():
    ufo_theta[k] = pd.DataFrame.from_dict(ufo_theta[k])


figure(figsize = (8, 6))
for i, e in enumerate(ufo_theta.keys()):
    subplot(1,2,i+1)

    tmp = ufo_theta[e]
    tmp = tmp - tmp.mean(0)
    tmp = tmp / tmp.std(0)
    plot(tmp, color = 'grey', alpha=0.5)
    plot(tmp.mean(1), color = 'red', linewidth = 3)

    title(e)

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Corr_UFO_Theta.png"))
show()

# # ax = subplot(211)
# plot(lfp.restrict(wake_ep))
# plot(lfp_filt.restrict(wake_ep))
# # subplot(212, sharex = ax)
# plot(power.restrict(wake_ep))
# show()