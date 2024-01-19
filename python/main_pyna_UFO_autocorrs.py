# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-19 10:35:20

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec
from itertools import combinations
# from functions import *
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

# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')])

ac_wake = []
ac_sws = []
ac_rem = []

for s in datasets:

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
    ufo_ep, ufo_ts = loadUFOs(path)

    if ufo_ts is not None:
        ufo_gr = nap.TsGroup({0:ufo_ts})

        ac_wake.append(nap.compute_autocorrelogram(ufo_gr,0.2, 5, ep=wake_ep, norm=False))
        ac_rem.append(nap.compute_autocorrelogram(ufo_gr, 0.2, 5, ep=rem_ep, norm=False))
        ac_sws.append(nap.compute_autocorrelogram(ufo_gr, 0.2, 5, ep=sws_ep, norm=False))

ac_wake = pd.concat(ac_wake, 1)
ac_rem = pd.concat(ac_rem, 1)
ac_sws = pd.concat(ac_sws, 1)

figure(figsize = (18, 6))

subplot(131)
plot(ac_wake, color='grey', alpha=0.8, linewidth=0.8)
plot(ac_wake.mean(1), linewidth=4)
ylim(0, 2)
title("wake")
subplot(132)
plot(ac_rem, color='grey', alpha=0.8, linewidth=0.8)
plot(ac_rem.mean(1), linewidth=4)
ylim(0, 2)
title("rem")
subplot(133)
plot(ac_sws, color='grey', alpha=0.8, linewidth=0.8)
plot(ac_sws.mean(1), linewidth=4)
ylim(0, 2)
title("sws")

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Auto_corrs_UFO.png"))
show()







