# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-07-13 16:08:51
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-07-13 16:59:37

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
from ufo_detection import *
from scipy import signal
import functools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

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

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


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

        ###############################################################################################
        # MEMORY MAP
        ###############################################################################################
        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel

        sign_channels = channels[ufo_channels[s][0]]

        filename = data.basename + ".dat"
                
        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)            

        lfp = nap.TsdFrame(t=timestep, d=fp)

        ep = nap.IntervalSet(start=position.time_support[0,0]+5,
            end=position.time_support[0,1]-5)

        ufo_ts = ufo_ts.restrict(ep)

        ang_vel = computeAngularVelocity(position['ry'], ep=position.time_support, bin_size=0.5)


        a = ufo_ts.value_from(position['ry'], position.time_support)
        b = nap.Ts(ufo_ts.t + 0.5).value_from(position['ry'], position.time_support)

        Y = ((b.d - a.d)>0.0)*1

        X = []
        for i, t in enumerate(ufo_ts.t):
            X.append(lfp.get(t-0.005, t+0.005)[:,sign_channels].d.flatten())
        
        X = np.array(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0
            ).fit(X_train, y_train)

        clf.predict(X_test)