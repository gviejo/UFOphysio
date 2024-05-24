# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-12 15:11:12

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
sys.path.append("../")
from functions import *
# import pynacollada as pyna
from ufo_detection import *
import yaml

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

alldatasets = yaml.safe_load(open(os.path.join(data_directory,'datasets_SOUND.yaml'), 'r'))

datasets = alldatasets["light_10s"]

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}

ccs = {i:[] for i in range(3)}

peths = {}

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
    sleep_ep = data.epochs['sleep']
    sws_ep = data.read_neuroscope_intervals('sws')    
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]
    
    

    ufo_ep, ufo_tsd = loadUFOs(path)

    analogin, ts = get_memory_map(os.path.join(data.path, data.basename+"_0_analogin.dat"), 2, frequency=20000)

    analogin = nap.Tsd(ts, np.abs(analogin[:,0].astype(int)))

    stim_ep = analogin.threshold(4000).time_support.merge_close_intervals(1.0)

    # pwr = np.array([np.mean(analogin.get(e[0,0], e[0,1])) for e in stim_ep])
    # idx = np.digitize(pwr, np.arange(0, 13000, 4000))

    sys.exit()
        
    cc_1 = nap.compute_eventcorrelogram(
        nap.TsGroup({0:ufo_tsd}), 
        stim_ep.starts, 0.01, 1, 
        # ep = sws_ep
        )

    cc_2 = nap.compute_eventcorrelogram(
        nap.TsGroup({0:ufo_tsd}), 
        stim_ep.starts, 0.01, 1, 
        ep = sleep_ep.set_diff(sws_ep)
        )

    figure()
    plot(cc_1, label='sws')
    plot(cc_2, label='wake')
    legend()
    show()

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    sign_channels = channels[0]
    ctrl_channels = channels[2]
    filename = data.basename + ".dat"    
    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

    # nSS = compute_meanNSS(fp, sign_channels, ctrl_channels, timestep)
    # nSS = nSS.bin_average(1/5000)
    # nSS.save(os.path.join(data.path, "nSS"))
    # # sys.exit()
        


    # peth = nap.compute_perievent(
    #     nap.TsGroup({0:ufo_tsd.restrict(sws_ep)}), 
    #     ttl.restrict(sws_ep), 0.1)

    # Writing for neuroscope
    peaks2 = stim_ep.starts.as_units('ms').index.values
    datatowrite = peaks2
    n = len(peaks2)
    texttowrite = np.repeat(np.array(['LIGHT 1']), n)    
    evt_file = os.path.join(data.path, data.basename+'.evt.py.lgt')
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()

for i in ccs.keys():
    ccs[i] = pd.concat(ccs[i], 1)

figure()
for i in ccs.keys():
    subplot(1,3,i+1)
    plot(ccs[i])
show()