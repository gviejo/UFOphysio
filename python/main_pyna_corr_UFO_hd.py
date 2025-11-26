# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-05 10:16:41
# %%
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions.functions import *
# import pynacollada as pyna
from ufo_detection import *

# %%
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
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


tuning_curves_ufo = {}

# %%
for s in datasets:
# for s in ['LMN-ADN/A5044/A5044-240402A']:
    print(s)

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
    # swr_ep, swr_ts = loadRipples(path)

    ufo_tsd = nap.load_file(os.path.join(path, data.basename + '_ufo_tsd.npz'))

    tokeep = ufo_tsd.restrict(wake_ep).d>6
    ufo_ts = ufo_tsd.restrict(wake_ep)[tokeep]

    if ufo_ts is not None:
        ufo_gr = nap.TsGroup({0:ufo_ts})

        if ufo_gr.rate[0] > 0.75:

            ep = position[['x', 'z']].time_support.loc[[0]]

            shifts = (np.arange(0, 2, 0.2) / np.mean(np.diff(position['ry'].t))).astype(int)

            # Tuning curves of the UFO during wake
            tmp = []
            for shift in shifts:
                pos_shifted = nap.Tsd(
                    position['ry'].t,
                    np.roll(position['ry'].values, shift),
                    time_support=position['ry'].time_support,
                )
                tc = nap.compute_tuning_curves(
                    ufo_gr,
                    pos_shifted,
                    bins=20,
                    range=(0, 2 * np.pi),
                    epochs=ep,
                    return_pandas=True,
                )
                tmp.append(tc)
            tmp = pd.concat(tmp, axis=1)
            tmp.columns = shifts
            tuning_curves_ufo[s] = tmp

figure()
gs = GridSpec(20, len(shifts))
keys = list(tuning_curves_ufo.keys())

for i in range(20):
    for j, shift in enumerate(shifts):
        ax = subplot(gs[i, j], projection='polar')
        plot(tuning_curves_ufo[keys[i]].iloc[:,j])
        xticks([])
        yticks([])

tight_layout()
show()
