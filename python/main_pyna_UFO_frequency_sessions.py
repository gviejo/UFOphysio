# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-12-01 12:31:17

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
# from functions import *
# import pynacollada as pyna
from ufo_detection import *
from matplotlib.pyplot import *
import nwbmatic as ntm

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
    ])


datasets = np.unique(datasets)

animals = np.unique([np.array([s.split("/")[1] for s in datasets])])

sessions = {a:[] for a in animals}

for s in datasets:
    sessions[s.split("/")[1]].append(s)

rates = {}


for a in animals:

    rates[a] = pd.DataFrame(columns = ['wak', 'rem', 'sws', 'intervals'],
        index = sessions[a],
        data = np.nan)      

    for s in sessions[a]:
        print(s)
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, s)
        basename = os.path.basename(path)
        filepath = os.path.join(path, "kilosort4", basename + ".nwb")

        if os.path.exists(filepath):
            nwb = nap.load_file(filepath)

            epochs = nwb['epochs']
            wake_ep = epochs[epochs.tags == "wake"]
            sws_ep = nwb['sws']
            rem_ep = nwb['rem']

            ufo_ep, ufo_ts = loadUFOs(path)
            
            if ufo_ts is not None:

                for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):

                    rates[a].loc[s, e] = ufo_ts.restrict(ep).rate


figure()
gs = GridSpec(1, 3)
for i, e in enumerate(['wak', 'rem', 'sws']):
    subplot(gs[0, i])
    for a in animals:
        data = rates[a][e].dropna().values
        if len(data) > 1:
            plot(np.arange(len(data)), data, 'o-')
    title(e)
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_across_sessions.pdf"))

    
