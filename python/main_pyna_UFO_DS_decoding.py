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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN_DG.list'), delimiter = '\n', dtype = str, comments = '#')



# for s in datasets:
for s in ["ADN-HPC/B5100/B5102/B5102-250917"]:

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
    ds_ep, ds_ts = loadDentateSpikes(path)

    nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
    ufo_ts = ufo_ts.value_from(nSS)
    ufo_ts = ufo_ts[ufo_ts > 5]

    if ufo_ts is not None:
        tuning_curves = nap.compute_tuning_curves(
            spikes, position['ry'], 60, range=(0, 2 * np.pi), epochs=position.time_support
        )

        SI = nap.compute_mutual_information(tuning_curves)

        spikes.set_info(SI=SI["bits/spike"])

        adn_spikes = spikes[(spikes.location == "adn") & (spikes.SI > 0.2)]
        adn_tuning_curves = tuning_curves.sel(unit=adn_spikes.keys())
        adn_spikes.peak = adn_tuning_curves.idxmax(dim="0").values

        ep = sws_ep[24]

        decoded, P = nap.decode_bayes(
            adn_tuning_curves,
            data=adn_spikes,
            epochs = ep,
            bin_size=0.02,
            sliding_window_size=3,
            uniform_prior=False
        )

figure()
ax = subplot(211)
[axvspan(s, e, color='blue', alpha=0.2) for s, e in ds_ep.intersect(ep).values]
plot(adn_spikes.to_tsd("peak").restrict(ep), '|k', ms=10)
[axvline(t, color='r', lw=1) for t in ufo_ts.restrict(ep).t]


ax2 = subplot(212, sharex=ax)
imshow(P.values.T, aspect='auto', origin='lower', extent=[ep.start[0], ep.end[0], 0, 2 * np.pi], cmap='jet')
[axvline(t, color='r', lw=2) for t in ufo_ts.restrict(ep).t]
show()
