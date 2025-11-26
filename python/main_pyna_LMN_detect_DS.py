# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-10-23 11:18:41

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
# import pynacollada as pyna
from ds_detection import *
from functions.functions import loadXML
import warnings
warnings.filterwarnings("ignore")

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN_DG.list'), delimiter = '\n', dtype = str, comments = '#')

ds_channels = np.genfromtxt(os.path.join(data_directory, 'channels_DS.txt'), delimiter =' ', dtype = str, comments ='#')
ds_channels = {a[0]: a[1:].astype('int') for a in ds_channels}


for s in datasets:
# for s in ["ADN-HPC/B5100/B5101/B5101-250502"]:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    #sws_ep = data.read_neuroscope_intervals('sws')

    # ufo_ep, ufo_ts = loadUFOs(path)
    #
    # if s not in ufo_channels.keys():
    #     print("No UFO channels specified for this session {}".format(s))
    #     continue

    if True:

        ###############################################################################################
        # MEMORY MAP
        ###############################################################################################
        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel
        # num_channels, fs, shank_to_channel, shank_to_keep = loadXML(path)

        # sign_channels = channels[ds_channels[s][0]]
        # ctrl_channels = channels[ds_channels[s][1]]
        filename = data.basename + ".eeg"

        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels, 1250)
        
        ds_ep, ds_tsd, nSS = detect_dentate_spikes(fp, ds_channels[s], timestep)

        
        ###########################################################################################################
        # Saving with pynapple
        ds_ep.save(os.path.join(path, data.basename + '_ds_ep'))
        ds_tsd.save(os.path.join(path, data.basename + '_ds_tsd'))

        ###########################################################################################################
        # Writing for neuroscope
        start = ds_ep.as_units('ms')['start'].values
        peaks = ds_tsd.as_units('ms').index.values
        ends = ds_ep.as_units('ms')['end'].values

        datatowrite = np.vstack((start,peaks,ends)).T.flatten()

        n = len(ds_ep)

        texttowrite = np.vstack(((np.repeat(np.array(['DS start 1']), n)),
                                (np.repeat(np.array(['DS peak 1']), n)),
                                (np.repeat(np.array(['DS stop 1']), n))
                                    )).T.flatten()
        
        evt_file = os.path.join(path, data.basename + '.evt.py.dsp')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()

        # sys.exit()
