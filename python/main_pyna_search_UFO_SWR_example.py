# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-27 12:35:03
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-29 11:05:48
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
from functions import *
import pynacollada as pyna
from ufo_detection import *
#%%
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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#')

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}



#%%
for s in datasets[[0]]:
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
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)
    rip_ep, rip_ts = loadRipples(path)

    ###############################################################################################
    # MEMORY MAP
    ###############################################################################################
    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    ufo_channels = channels[ufo_channels[s][0]]
    ca1_channels = channels[0]
    filename = data.basename + ".dat"    
    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

    pwr_ca1 = compute_power(fp, timestep, ca1_channels[[3]], [80, 200], frequency=20000, wsize=41)
    
    pwr_ufo = compute_power(fp, timestep, ufo_channels[::3], [80, 200], frequency=20000, wsize=41)
    
    #%%
    t = 1913.856
    ep = nap.IntervalSet(t-120, t+120)    

    figure()
    axvline(t)
    for pwr, name in zip([pwr_ufo, pwr_ca1], ['ufo', 'ca1']):
        tmp = pwr.bin_average(0.1).smooth(3.0).restrict(ep)
        tmp = tmp - np.mean(tmp)
        tmp = tmp / np.std(tmp)
        plot(tmp, label=name)
    legend()
    show()
    # %%

    pwrs = pd.DataFrame.from_dict({"ca1":pwr_ca1.bin_average(0.1).as_series(),"ufo":pwr_ufo.bin_average(0.1).as_series()})
    
    pwrs = nap.TsdFrame(pwrs)

    # %%
    # stft
    f, tt, ZZ = compute_spectrogram(fp, timestep, 3, frequency=20000,time_units='s')
    idx = f<200
    Z = nap.TsdFrame(t=tt, d=ZZ[idx].T)        
    lfp = nap.Tsd(t=timestep, d=fp[:,3])

    figure(figsize=(12, 6))
    subplot(211)    
    plot(lfp.get(t-1, t+1))
    xlim(t-1, t+1)
    subplot(212)
    imshow(Z.get(t-1, t+1).d.T, cmap = 'jet', origin='lower', aspect='auto', extent=(t-1, t+1, 0, f[idx][-1]))
    show()


# %%
pwrs.save(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/pwr_ufo_ca1.npz"))



# %%
