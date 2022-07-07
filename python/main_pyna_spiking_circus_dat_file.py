# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-20 11:33:35

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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

# 53 1 73

folder = '/mnt/Data2/SpykingCircus/LMN'


for s in ['LMN-ADN/A5002/A5002-200303B']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]

    ###############################################################################################
    # MEMORY MAP
    ###############################################################################################
    data.load_neurosuite_xml(data.path)
    gr = np.unique(spikes._metadata["group"].values)[0]
    channels = data.group_to_channel[gr]

    if len(np.unique(spikes._metadata["group"].values)) > 1:
        sys.exit()

    filename = data.basename + ".dat"    
            
    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)
    
    ##############################################################################################
    # WRITING
    ##############################################################################################
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder2 = os.path.join(folder, data.basename.split('-')[0])
    if not os.path.exists(folder2):
        os.mkdir(folder2)
    folder3 = os.path.join(folder2, data.basename)
    if not os.path.exists(folder3):
        os.mkdir(folder3)

    filepath = os.path.join(folder3, data.basename + '_sh'+str(gr)+'.dat')
    
    f = open(filepath, 'wb')

    for i in range(len(fp)):
        for j, c in enumerate(channels):
            f.write(fp[i,c])
    f.close()

