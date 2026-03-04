# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-12-10 12:29:28
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

ahv_ufo = {}
lin_ufo = {}

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

        if ufo_gr.rate[0] > 0.1:

            ep = position[['x', 'z']].time_support.loc[[0]]

            shifts_t = np.arange(-2, 2, 0.04)
            shifts = (shifts_t / np.mean(np.diff(position.t))).astype(int)[::-1]

            # linear speed
            # lin = computeLinearSpeed(position[['x', 'z']], ep, 0.02)
            lin = computeLinearVelocity(position[['x', 'z']], ep, 0.02)
            shifts_t = np.arange(-2, 2, 0.04)
            shifts = (shifts_t / np.mean(np.diff(lin.t))).astype(int)[::-1]

            tmp = []
            for shift in shifts:
                lin_shifted = nap.Tsd(
                    lin.t,
                    np.roll(lin.values, shift),
                    time_support=lin.time_support,
                )
                

                lintc = nap.compute_tuning_curves(
                    ufo_gr,
                    lin_shifted,
                    bins=[[-3, -0.025, 0.025, 3]],
                    # bins=[[-4, 0, 4]],
                    # range = (-1, 1),
                    epochs=ep,
                    return_pandas=True,
                )
                tmp.append(lintc)
                lintc = lintc/lintc.sum(0)

            tmp = pd.concat(tmp, axis=1)
            tmp.columns = shifts_t         
            lin_ufo[s] = tmp


            # ahv tuning
            ahv = computeAngularVelocity(position['ry'], ep, 0.02)
            shifts_t = np.arange(-2, 2, 0.04)
            shifts = (shifts_t / np.mean(np.diff(ahv.t))).astype(int)[::-1]

            tmp = []
            for shift in shifts:
                ahv_shifted = nap.Tsd(
                    ahv.t,
                    np.roll(ahv.values, shift),
                    time_support=ahv.time_support,
                )
                

                ahvtc = nap.compute_tuning_curves(
                    ufo_gr,
                    ahv_shifted,
                    bins=[[-3, -0.025, 0.025, 3]],
                    # bins=[[-4, 0, 4]],
                    # range = (-1, 1),
                    epochs=ep,
                    return_pandas=True,
                )
                tmp.append(ahvtc)
                ahvtc = ahvtc/ahvtc.sum(0)

            tmp = pd.concat(tmp, axis=1)
            tmp.columns = shifts_t         
            ahv_ufo[s] = tmp


# ahv_ufo = pd.DataFrame.from_dict(ahv_ufo)


# figure()
# # plot(ahv_ufo, alpha=0.5)
# plot(ahv_ufo.mean(1), color='k', linewidth=3)
# fill_between(ahv_ufo.index, ahv_ufo.mean(1)-ahv_ufo.std(1), ahv_ufo.mean(1)+ahv_ufo.std(1), color='k', alpha=0.2)

# show()
# figure()
# count = 0
# for 
# cmap = plt.cm.viridis  # or "plasma", "jet", etc.

# figure()
# for i, s in enumerate(ahv_ufo.keys()):
#     ax = subplot(6, 7, i+1)
#     n = len(ahv_ufo[s].columns)                       # number of lines
#     colors = cmap(np.linspace(0, 1, n))    
#     for j, sh in enumerate(ahv_ufo[s].columns):
#         plot(ahv_ufo[s][sh], color=colors[j])
#     # imshow(ahv_ufo[s].values.T, aspect='auto', origin='lower', cmap='jet')
# show()

figure()
for k in range(ahv_ufo[s].shape[0]):
    tmp = np.array([ahv_ufo[s].iloc[k].values for i, s in enumerate(ahv_ufo.keys())])
    plot(shifts_t, tmp.mean(0), label=ahv_ufo[s].index[k])
    fill_between(shifts_t, tmp.mean(0)-tmp.std(0)/tmp.shape[0], tmp.mean(0)+tmp.std(0)/tmp.shape[0], alpha=0.3)
    legend()    
# show()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/figUFO_AHV_tc_offset.pdf"))



# figure()
# index = np.logical_and(shifts_t>0, shifts_t<0.5)
# tokeep = shifts_t[index]

# gs = GridSpec(10, len(tokeep))
# keys = list(tuning_curves_ufo.keys())[0:10]

# for i in range(len(keys)):
#     for j, shift in enumerate(tokeep):
#         ax = subplot(gs[i, j], projection='polar')
#         plot(tuning_curves_ufo[keys[i]][shift])
#         xticks([])
#         yticks([])

# tight_layout()
# show()
