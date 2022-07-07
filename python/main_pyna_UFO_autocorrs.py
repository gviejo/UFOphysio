# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-20 12:42:44

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

ac_wake = []
ac_sws = []

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
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)

    ufo_gr = nap.TsGroup({0:ufo_ts})

    ac_wake.append(nap.compute_autocorrelogram(ufo_gr, 0.1, 3, ep=wake_ep))
    ac_sws.append(nap.compute_autocorrelogram(ufo_gr, 0.1, 3, ep=sws_ep))

ac_wake = pd.concat(ac_wake, 1)
ac_sws = pd.concat(ac_sws, 1)

figure()

subplot(221)
plot(ac_wake)
title("wake")
subplot(222)
plot(ac_sws)
title("sws")
subplot(223)
plot(ac_wake.mean(1))
title("wake")
subplot(224)
plot(ac_sws.mean(1))
title("sws")

show()







