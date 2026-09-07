# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-07-13 16:08:51
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-12-03 11:32:22

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os

from matplotlib import pyplot as plt
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations

from sklearn.decomposition import KernelPCA
from matplotlib.colors import hsv_to_rgb
sys.path.append("..")
from functions import *
from ufo_detection import *
from scipy import signal
import functools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pynaviz import scope

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

mdpec = {}

# for s in datasets:
for s in ["LMN-ADN/A5044/A5044-240401A"]:

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

    nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
    ufo_ts = ufo_ts.value_from(nSS)
    ufo_ep = ufo_ep[ufo_ts > 5]
    ufo_ts = ufo_ts[ufo_ts > 5]

    spikes = spikes[(spikes.location=="adn") | (spikes.location=="lmn")]
    spikes = spikes.getby_threshold("rate", 1)
    tuning_curves = nap.compute_tuning_curves(
        spikes, position['ry'], 60, range=(0, 2 * np.pi), epochs=position.time_support
    )
    SI = nap.compute_mutual_information(tuning_curves)
    spikes.set_info(SI=SI["bits/spike"])
    spikes.set_info(SI=SI["bits/spike"])
    spikes = spikes[spikes.SI > 0.3]

    if len(spikes)>8:

        tuning_curves = tuning_curves.sel(unit=spikes.keys())
        spikes.peak = tuning_curves.idxmax(dim="0").values
        spikes.order = np.argsort(np.argsort(spikes.peak.values))

        decoded, P = nap.decode_bayes(
            tuning_curves,
            data=spikes,
            epochs = sws_ep,
            bin_size=0.004,
            sliding_window_size=3,
            uniform_prior=True
        )
        # decoded, P = nap.decode_template(
        #     tuning_curves,
        #     data=spikes,
        #     epochs = wake_ep,
        #     bin_size=0.01,
        #     sliding_window_size=3,
        #     metric="correlation",
        # )


        decoded2 = nap.Tsd(t=decoded.t, d=np.unwrap(decoded.values), time_support=decoded.time_support)
        angspeed = decoded2.derivative().abs()#.smooth(0.02)

        # dpec = nap.compute_perievent_continuous(angspeed, ufo_ts.restrict(sws_ep), (-0.3, 0.3), ep=sws_ep)
        #
        # mdpec[s] = np.nanmean(dpec, axis=1)

        scope(globals(), layout_path = '/mnt/home/gviejo/UFOphysio/python/decoding/layout_2026-03-05_15-54.json')

        # # Tuning curves
        # figure()
        # g = tuning_curves.plot(
        #     row="unit",
        #     col_wrap=5,
        #     subplot_kws={"projection": "polar"},
        #     sharey=False
        # )
        # xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        # g.set_titles("")
        # g.set_xlabels("")
        #
        #
        # # Raster
        # figure()
        # cmap = plt.get_cmap("hsv")
        # ep = sws_ep
        # for i, (n,o,p) in enumerate(zip(spikes.index, spikes.order, spikes.peak)):
        #     neuron_spikes = spikes[n].restrict(ep).t
        #     h = (p % (2 * np.pi)) / (2 * np.pi)
        #     plot(neuron_spikes, np.ones(len(neuron_spikes)) * o, '|', ms =20, mew = 5,
        #          color = hsv_to_rgb([h,1,1])
        #          )
        # [axvline(t, color='r', lw=1) for t in ufo_ts[(spikes.to_tsd().count(ep=ufo_ep)>4).values].restrict(ep).t]
        # show()



