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

eta_angv = {}
eta_linv = {}

ufo_vel = []

ang_lin = []

trans_start = {}
trans_stop = {}

peth_start = {}
peth_stop = {}

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

        ep = position[['x', 'z']].time_support.loc[[0]]

        bin_size = 0.05        
        lin_velocity = computeLinearVelocity(position[['x', 'z']], ep, bin_size)
        lin_velocity = lin_velocity*100.0
        ang_velocity = computeAngularVelocity(position['ry'], ep, bin_size)
        ang_velocity = np.abs(ang_velocity)

        lin_corr = nap.compute_event_trigger_average(ufo_gr, lin_velocity, bin_size, (-5.0, 5.0), ep)
        ang_corr = nap.compute_event_trigger_average(ufo_gr, ang_velocity, bin_size, (-1.0, 1.0), ep)

        eta_linv[s] = lin_corr.loc[0].as_series()
        eta_angv[s] = ang_corr.loc[0].as_series()
        
        # Search for transition
        tmp = ang_velocity.threshold(0.025, "below").time_support.drop_short_intervals(2)

        # Starts
        trans_start[s] = nap.compute_eventcorrelogram(ufo_gr, tmp.ends, 0.01, 2, norm=True)[0]
        trans_stop[s] = nap.compute_eventcorrelogram(ufo_gr, tmp.starts, 0.01, 2, norm=True)[0]

        peth_start[s] = nap.compute_perievent(ufo_gr, tmp.ends, 2)[0].to_tsd().as_series()
        peth_stop[s] = nap.compute_perievent(ufo_gr, tmp.starts, 2)[0].to_tsd().as_series()
                
        # Tuning curves of the UFO during wake
        tuning_curves_ufo[s] = nap.compute_tuning_curves(
            ufo_gr,
            position['ry'],
            bins=30,
            range=(0, 2 * np.pi),
            epochs=ep,
            return_pandas=True,
        )[0]


# %%
eta_linv = pd.DataFrame.from_dict(eta_linv)
eta_angv = pd.DataFrame.from_dict(eta_angv)

trans_start = pd.DataFrame.from_dict(trans_start)
trans_stop = pd.DataFrame.from_dict(trans_stop)


# %%
datatosave = {
    "eta_linv":eta_linv, "eta_angv":eta_angv,
    "trans_start":trans_start, "trans_stop":trans_stop,
    "peth_start":peth_start, "peth_stop":peth_stop
    }

import _pickle as cPickle
cPickle.dump(datatosave, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CORR_UFO_SPEED.pickle"), 'wb'))

# %%
figure()
subplot(211)
tmp = trans_start.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=2)
# plot(tmp.loc[-0.4:0.4], alpha = 0.5, linewidth = 1)
plot(tmp.loc[-0.4:0.4].mean(1), label = "pause -> run", linewidth = 4)
m = tmp.loc[-0.4:0.4].mean(1)
s = tmp.loc[-0.4:0.4].std(1).values
fill_between(m.index.values, m.values - s, m.values + s, alpha = 0.5)
axvline(0.0)
maxv = s.max()
ylim(0, maxv+1)
legend()
subplot(212)
tmp = trans_stop.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=2)
# plot(tmp.loc[-0.4:0.4], alpha = 0.5, linewidth = 1)
plot(tmp.loc[-0.4:0.4].mean(1), label = "run -> pause", linewidth = 4)
m = tmp.loc[-0.4:0.4].mean(1)
s = tmp.loc[-0.4:0.4].std(1).values
fill_between(m.index.values, m.values - s, m.values + s, alpha = 0.5)
axvline(0.0)
legend()
ylim(0, maxv+1)



# %%
figure(figsize = (8, 6))

gs = GridSpec(1,2)

subplot(gs[0,0])
tmp = eta_linv - eta_linv.mean()
tmp = tmp / tmp.std()
plot(tmp, color='grey', alpha=0.8, linewidth=0.8)
plot(tmp.mean(1), linewidth=4)
title("Linear speed")
axvline(0.0)
ylabel("Z")

subplot(gs[0,1])
tmp = eta_angv - eta_angv.mean()
tmp = tmp / tmp.std()
plot(tmp, color='grey', alpha=0.8, linewidth=0.8)
plot(tmp.mean(1), linewidth=4)
title("Angular speed")
axvline(0.0)
ylabel("Z")

show()

figure()
for i, s in enumerate(tuning_curves_ufo.keys()):
    subplot(6, 11, i+1, projection='polar')
    plot(tuning_curves_ufo[s])
show()


