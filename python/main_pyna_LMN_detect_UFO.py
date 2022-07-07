# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-06-14 11:35:30

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *

############################################################################################### 
# GENERAL infos
###############################################################################################
#data_directory = '/mnt/DataGuillaume/'
data_directory = '/mnt/Data2/'
#datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#')
#datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_MMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = ['LMN-PSB-2/A3018/A3018-220613A']

infos = getAllInfos(data_directory, datasets)

# 53 1 73

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    #sws_ep = data.read_neuroscope_intervals('sws')    
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]

    
    ufo_ep, ufo_ts = loadUFOs(path)

    if ufo_ep is None:    
        print(s)
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
        ###############################################################################################
        # MEMORY MAP
        ###############################################################################################
        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel[np.unique(spikes._metadata["group"].values)[0]]    
        filename = data.basename + ".dat"    
                
        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

        # sys.exit()

        ufo_ep, ufo_tsd = detect_ufos(fp, channels, timestep)

        
        ###########################################################################################################
        # Writing for neuroscope
        start = ufo_ep.as_units('ms')['start'].values
        peaks = ufo_tsd.as_units('ms').index.values
        ends = ufo_ep.as_units('ms')['end'].values

        datatowrite = np.vstack((start,peaks,ends)).T.flatten()

        n = len(ufo_ep)

        texttowrite = np.vstack(((np.repeat(np.array(['UFO start 1']), n)), 
                                (np.repeat(np.array(['UFO peak 1']), n)),
                                (np.repeat(np.array(['UFO stop 1']), n))
                                    )).T.flatten()

        #evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.ufo'
        evt_file = os.path.join(path, data.basename + '.evt.py.ufo')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()   


