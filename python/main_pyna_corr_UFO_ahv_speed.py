# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-12-10 14:07:53
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

tcahv = {}
tclin = {}
tcahvlin = {}


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

    tokeep = ufo_tsd.restrict(wake_ep).d>5
    ufo_ts = ufo_tsd.restrict(wake_ep)[tokeep]

    if ufo_ts is not None:
        ufo_gr = nap.TsGroup({0:ufo_ts})

        if ufo_gr.rate[0] > 0.01:

            ep = position[['x', 'z']].time_support[0]

            lin = computeLinearVelocity(position[['x', 'z']], ep, 0.02)*100
            ahv = computeAngularVelocity(position['ry'], ep, 0.02)

            
            tclin[s] = nap.compute_tuning_curves(
                ufo_gr,
                lin,
                bins=20,
                range = (0, 0.4),
                epochs=ep,
                return_pandas=True,
            )[0]

            tcahv[s] = nap.compute_tuning_curves(
                    ufo_gr,
                    ahv,
                    bins=30,
                    range = (-4, 4),
                    epochs=ep,
                    return_pandas=True,
                )[0]

            tmp = nap.TsdFrame(
                t = lin.t, d = np.vstack((lin, ahv)).T, columns = ["lin", "ahv"]
                )

            tcahvlin[s] = nap.compute_tuning_curves(
                    ufo_gr,
                    tmp,
                    bins=[20, 30],
                    # bins=[[-4, 0, 4]],
                    # range = (-1, 1),
                    epochs=ep,                    
                )


tclin = pd.concat(tclin, axis=1) 
tcahv = pd.concat(tcahv, axis=1) 
tcahvlin_ = np.nanmean(np.stack([tcahvlin[s].values[0] for s in tcahv.keys()]), 0)

figure(figsize = (9,3))
gs = GridSpec(1,3)

subplot(gs[0,0])
plot(tclin.mean(1))
title("Linear speed")

subplot(gs[0,1])
plot(tcahv.mean(1))
title("Angular head velocity")

subplot(gs[0,2])
imshow(tcahvlin_.T, aspect='auto', origin='lower', cmap='jet')

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/figUFO_AHV_LIN_tc.pdf"), dpi=300)