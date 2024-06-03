# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-29 18:13:31

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions import *
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

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])

eta_angv = {}
eta_linv = {}

ufo_vel = []

ang_lin = []

for s in datasets:
# for s in ['LMN/A1411/A1411-200910A']:

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

        lin_corr = nap.compute_event_trigger_average(ufo_gr, lin_velocity, bin_size, (-10.0, 10.0), ep)
        ang_corr = nap.compute_event_trigger_average(ufo_gr, ang_velocity, bin_size, (-1.0, 1.0), ep)

        eta_linv[s] = lin_corr.loc[0].as_series()
        eta_angv[s] = ang_corr.loc[0].as_series()

        # 2d histogram

        bin_size = 0.5
        n_bins = 50

        lin_velocity = computeLinearVelocity(position[['x', 'z']], ep, bin_size)
        lin_velocity = lin_velocity*100.0
        ang_velocity = computeAngularVelocity(position['ry'], ep, bin_size)
        ang_velocity = np.abs(ang_velocity)

        
        # feature = nap.TsdFrame(t=lin_velocity.t, d=np.vstack((lin_velocity.d, ang_velocity.d)).T)
        # tc, xybins = nap.compute_2d_tuning_curves(ufo_gr, feature, n_bins, ep, (0.0, 5.0, 0.0, np.pi))


        # geomspace
        xb = np.geomspace(0.01, 10.0, n_bins) # linaer
        yb = np.geomspace(0.01, 2*np.pi, n_bins) # angular

        idx = np.vstack((
            np.digitize(lin_velocity.d, xb),
            np.digitize(ang_velocity.d, yb)
            )).T

        count = ufo_gr.count(ep, bin_size).d.flatten()

        hist_ufo_vel = np.zeros((n_bins, n_bins))
        for i, (j,k) in enumerate(idx):
            hist_ufo_vel[j-1,k-1] += count[i]

        hist_ang_lin = np.zeros((n_bins, n_bins))
        for i, (j,k) in enumerate(idx):
            hist_ang_lin[j-1,k-1] += 1.0
        hist_ang_lin /= np.sum(hist_ang_lin)
        ang_lin.append(hist_ang_lin)

        hist_ufo_vel = hist_ufo_vel/(hist_ang_lin+1)
        ufo_vel.append(hist_ufo_vel)



eta_linv = pd.DataFrame.from_dict(eta_linv)
eta_angv = pd.DataFrame.from_dict(eta_angv)

ufo_vel = np.array(ufo_vel)
ang_lin = np.array(ang_lin)

datatosave = {"eta_linv":eta_linv, "eta_angv":eta_angv}
import _pickle as cPickle
cPickle.dump(datatosave, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CORR_UFO_SPEED.pickle"), 'wb'))




figure(figsize = (8, 6))

gs = GridSpec(2,2)

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

gs2 = GridSpecFromSubplotSpec(2, 2, gs[1,0], width_ratios=[0.8,0.4], height_ratios=[0.4,0.8])

subplot(gs2[0,0])
tmp1 = ufo_vel.mean(0).mean(0)[0:-1]
semilogx(xb[0:-1], tmp1/tmp1.sum())
tmp2 = ang_lin.mean(0).mean(0)[0:-1]
semilogx(xb[0:-1], tmp2/tmp2.sum())

subplot(gs2[1,0])
imshow(ufo_vel.mean(0)[0:-1,0:-1], cmap='jet', origin='lower', aspect='auto')
title("UFO")
xlabel("Linear (cm/s)")
ylabel("Angular (rad/s)")
xticks(np.arange(0, ufo_vel.shape[1], 10), np.round(xb[::10], 3))
yticks(np.arange(0, ufo_vel.shape[2], 10), np.round(yb[::10], 3))

subplot(gs2[1,1])
tmp1 = ufo_vel.mean(0).mean(1)[0:-1]
semilogy(tmp1/tmp1.sum(), yb[0:-1])
tmp2 = ang_lin.mean(0).mean(1)[0:-1]
semilogy(tmp2/tmp2.sum(), yb[0:-1])

subplot(gs[1,1])

imshow(ang_lin.mean(0)[0:-1,0:-1], cmap='jet', origin='lower', aspect='auto')
title("occupancy")
xlabel("Linear (cm/s)")
ylabel("Angular (rad/s)")
xticks(np.arange(0, ufo_vel.shape[1], 10), np.round(xb[::10], 3))
yticks(np.arange(0, ufo_vel.shape[2], 10), np.round(yb[::10], 3))


tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Corr_speed.png"))
show()







