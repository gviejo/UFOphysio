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

datasets = np.hstack([
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_ADN_DG.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


# datasets = [#"LMN-ADN/A5044/A5044-240401B",
            # "OPTO/B3000/B3007/B3007-240501A",
            # "OPTO/B3000/B3009/B3009-240502C",
            # "OPTO/B3000/B3010/B3010-240510C",
            # "LMN-ADN/A5044/A5044-240403B",
            # "OPTO/B3000/B3007/B3007-240502A",
            # "OPTO/B3000/B3009/B3009-240503C",
            # "OPTO/B3000/B3010/B3010-240511A"]


# for s in datasets[19:]:
# for s in ['LMN/A1411/A1411-200910A']:
# for s in ['ADN-HPC/B3214/B3218-241018']:
# for s in ["ADN-HPC/B5100/B5102/B5102-250915"]:
# for s in ["ADN-HPC/B5100/B5107/B5107-260218"]:
for s in [
    # "ADN-HPC/B5100/B5107/B5107-260217",
    "ADN-HPC/B5100/B5107/B5107-260218",
    "ADN-HPC/B5100/B5107/B5107-260219",
    "ADN-HPC/B5100/B5107/B5107-260224",
    "ADN-HPC/B5100/B5107/B5107-260227"
]:
# for s in datasets:

    ###############################################################################################
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    basename = data.basename
    wake_ep = data.epochs['wake']
    #sws_ep = data.read_neuroscope_intervals('sws')

    ufo_ep, ufo_ts = loadUFOs(path)

    if s not in ufo_channels.keys():
        print("No UFO channels specified for this session {}".format(s))
        break

    # if ufo_ep is None:
    if True:

        ###############################################################################################
        # MEMORY MAP
        ###############################################################################################
        data = nap.EphysReader(path, format="NeuroScopeIO")
        lfp = data[basename + ".dat"]
        metadata = lfp.metadata
        sign_channels = metadata[(metadata.group == ufo_channels[s][0]) & (metadata.skip == False)].index.values
        ctrl_channels = metadata[(metadata.group == ufo_channels[s][1]) & (metadata.skip == False)].index.values

        # sign_channels = sign_channels
        # ctrl_channels = ctrl_channels
        #
        ufo_ep, ufo_tsd, nSS = detect_ufos_v2(lfp.d, sign_channels, ctrl_channels, lfp.t, (4, 100))
        
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
        ufo_ep.save(os.path.join(path, basename + '_ufo_ep'))
        ufo_tsd.save(os.path.join(path, basename + '_ufo_tsd'))
        # nSS = nSS.bin_average(1/5000)
        nSS.save(os.path.join(path, "nSS_LMN"))


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
        
        evt_file = os.path.join(path, basename + '.evt.py.ufo')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()   


    # # pynaviz check
    data = nap.EphysReader(path, format="NeuroScopeIO")
    # nSS = nap.load_file(os.path.join(path, "nSS_LMN.npz"))
    nSS = nSS.bin_average(1/5000)
    from pynaviz import scope
    scope({
        "UFO": ufo_ep,
        "EEG": data[basename+".dat"],
        "nSS": nSS
    }, layout_path="layout_2026-05-21_14-52.json")