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
from ufo_detection import *

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
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


datasets = [#"LMN-ADN/A5044/A5044-240401B",
            # "OPTO/B3000/B3007/B3007-240501A",
            # "OPTO/B3000/B3009/B3009-240502C",
            # "OPTO/B3000/B3010/B3010-240510C",
            # "LMN-ADN/A5044/A5044-240403B",
            # "OPTO/B3000/B3007/B3007-240502A",
            # "OPTO/B3000/B3009/B3009-240503C",
            "OPTO/B3000/B3010/B3010-240511A"]

# for s in datasets[19:]:
# for s in ['LMN/A1411/A1411-200910A']:
# for s in datasets:
for s in ['ADN-HPC/B3214/B3218-241018']:
    print(s)
    sys.exit()
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

    # if ufo_ep is None:        
    if True:
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

        # # get spike time and clu from res/clu
        # clu = np.genfromtxt(os.path.join(path, s.split("/")[-1]+".clu."+str(ufo_channels[s][0]+1)), dtype="int")[1:]
        # res = np.genfromtxt(os.path.join(path, s.split("/")[-1]+".res."+str(ufo_channels[s][0]+1)), dtype="int")

        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)
        
        ufo_ep, ufo_tsd, nSS = detect_ufos_v2(fp, sign_channels, ctrl_channels, timestep)
        
        ############################
        # Higher threshold for wake
        from pynapple.core._core_functions import _restrict

        idx = _restrict(ufo_tsd.t, wake_ep.start, wake_ep.end)

        tokeep = np.ones(len(ufo_tsd), dtype=bool)
        for i in range(len(ufo_tsd)):
            if i in idx and ufo_tsd[i] < 7:
                tokeep[i]=False

        ufo_tsd = ufo_tsd[tokeep]
        ufo_ep = ufo_ep[tokeep]
        ####################
        
        # Saving with pynapple
        ufo_ep.save(os.path.join(path, data.basename + '_ufo_ep'))
        ufo_tsd.save(os.path.join(path, data.basename + '_ufo_tsd'))
        nSS = nSS.bin_average(1/5000)
        nSS.save(os.path.join(data.path, "nSS_LMN"))


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
        
        evt_file = os.path.join(path, data.basename + '.evt.py.ufo')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()   

        # sys.exit()
