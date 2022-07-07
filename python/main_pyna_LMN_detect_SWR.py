# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-19 14:16:39

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
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

rip_ch = {'A5022':3,'A5026':4, 'A5027':4}

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
    sleep_ep = data.epochs['sleep']

    lfp = data.load_lfp(channel=rip_ch[data.basename.split("-")[0]],extension='.eeg',frequency=1250.0)
    rip_ep, rip_tsd = pyna.eeg_processing.detect_oscillatory_events(
                                                lfp = lfp,
                                                epoch = sleep_ep,
                                                freq_band = (100,300),
                                                thres_band = (1, 10),
                                                duration_band = (0.02,0.2),
                                                min_inter_duration = 0.02
                                                )
    
    
    ###########################################################################################################
    # Writing for neuroscope
    start = rip_ep.as_units('ms')['start'].values
    peaks = rip_tsd.as_units('ms').index.values
    ends = rip_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,peaks,ends)).T.flatten()

    n = len(rip_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyRip start 1']), n)), 
                            (np.repeat(np.array(['PyRip peak 1']), n)),
                            (np.repeat(np.array(['PyRip stop 1']), n))
                                )).T.flatten()

    #evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.rip'
    evt_file = os.path.join(path, data.basename + '.evt.py.rip')
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()   

    
