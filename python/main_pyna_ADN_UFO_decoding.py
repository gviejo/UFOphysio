# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-02-06 20:38:40

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
from scipy import signal
from multiprocessing import Pool
import functools

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

angdist = {}

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
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)

    if ufo_ts is not None:        

        # idx = spikes._metadata[spikes._metadata["location"].str.contains("adn")].index.values
        spikes = spikes.getby_category("location")["adn"]
        
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves)    
        tcurves = tuning_curves
        SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

        spikes.set_info(SI)

        adn = list(spikes.getby_category("location")["adn"].getby_threshold("SI", 0.6).index)

        spikes = spikes[adn]
        tuning_curves = tuning_curves[adn]
        
        decoded, proba_feature = nap.decode_1d(
            tuning_curves=tuning_curves,
            group=spikes,
            ep=position['ry'].time_support.loc[[0]],
            bin_size=0.005,  # second
            feature=position['ry'],
        )

        ufo_wake = ufo_ts.restrict(position.time_support)

        if len(ufo_wake) > 10:
            tmp = nap.compute_perievent_continuous(position['ry'], ufo_ts, minmax=(-5, 5), ep=position.time_support)

            tmp2 = np.expand_dims(ufo_ts.value_from(decoded).values, 0)

            dist = np.abs((tmp - tmp2)).values
            dist[np.isnan(dist)] = 0.0
            dist = dist%(np.pi)

            angdist[s] = pd.Series(index=tmp.t, data=np.mean(dist, 1))



angdist = pd.DataFrame.from_dict(angdist)