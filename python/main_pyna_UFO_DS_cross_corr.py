# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-18 17:59:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-27 12:45:33
# %%
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpecFromSubplotSpec

from python.functions.functions import load_mean_waveforms
from ufo_detection import *
from matplotlib.pyplot import *

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

cc_short = {}
cc_long = {}
perievent = {}

cc_short2 = {}
cc_long2 = {}
perievent2 = {}

all_cc_ufo_dg = {}

ratios = []

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
    sleep_ep = data.epochs['sleep']
    sws_ep = data.read_neuroscope_intervals('sws')

    sws_ep = sws_ep.intersect(sleep_ep[0])

    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)

    # Waveform classification
    spikes.location = [v.lower() for v in spikes.location.values]
    meanwavef, maxch = load_mean_waveforms(path)
    spikes.maxch = np.hstack([chs for chs in maxch.values()])
    location = spikes.location.copy()
    location[(spikes.maxch < 30) & (spikes.location == "hpc")] = "ca1"
    location[(spikes.maxch >= 30) & (spikes.location == "hpc")] = "dg"
    spikes.location = location

    spikes = spikes[(spikes.location == 'adn') | (spikes.location == "dg") | (spikes.location == "ca1")]

    spikes = spikes[spikes.rate > 1.0]

    if ufo_ts is not None and ds_ts is not None:
        grp = nap.TsGroup({0:ufo_ts, 1:ds_ts, }, metadata={"evt":np.array(['ufo', 'ds'])})

        ufo_cc = nap.compute_crosscorrelogram(grp, 1, 2000, sws_ep, norm=True)
        cc_long[s] = ufo_cc[(0,1)]
        ufo_cc = nap.compute_crosscorrelogram(grp, 0.01, 0.5, sws_ep, norm=True)
        cc_short[s] = ufo_cc[(0,1)]
        perievent[s] = nap.compute_perievent(ds_ts, ufo_ts.restrict(sws_ep), minmax=(-100, 100)).to_tsd()

        # Reversed
        ufo_cc = nap.compute_crosscorrelogram(grp, 0.5, 100, sws_ep, norm=True, reverse=True)
        cc_long2[s] = ufo_cc[(1,0)]
        ufo_cc = nap.compute_crosscorrelogram(grp, 0.01, 1, sws_ep, norm=True, reverse=True)
        cc_short2[s] = ufo_cc[(1,0)]
        perievent2[s] = nap.compute_perievent(ufo_ts, ds_ts.restrict(sws_ep), minmax=(-100, 100)).to_tsd()

        # Spikes cross-corr
        all_cc_ufo_dg[s] = {}
        for state, ep in zip(['wake', 'sws'], [wake_ep, sws_ep]):
            cc = nap.compute_eventcorrelogram(spikes[spikes.group == 0], ufo_ts, 0.001, 0.1, ep, norm=True)
            cc = (cc - cc.mean(0)) / cc.std(0)
            all_cc_ufo_dg[s][state] = cc

        # %%
        # How many ufos are followed by a ds between 0-100ms?
        tmp = nap.compute_perievent(ds_ts, ufo_ts.restrict(sws_ep), minmax=(0, 0.1))
        count = np.sum(~np.isnan(tmp.rate))
        ratio = count/len(ufo_ts.restrict(sws_ep))

        # How many ds are preceded by a ufo between -50-5ms?
        tmp = nap.compute_perievent(ufo_ts, ds_ts.restrict(sws_ep), minmax=(-0.1, 0.0))
        count = np.sum(~np.isnan(tmp.rate))
        ratio2 = count/len(ds_ts.restrict(sws_ep))

        ratios.append(pd.DataFrame(
            index = [s.split("/")[-1]],
            data = {"ufo_to_ds": ratio,
                    "ds_to_ufo": ratio2,}
        ))


cc_long = pd.DataFrame.from_dict(cc_long)
cc_short = pd.DataFrame.from_dict(cc_short)

ratios = pd.concat(ratios)

# %%

ccs = {"long":cc_long, "short":cc_short, "perievent":perievent, "ratios":ratios}
import _pickle as cPickle
cPickle.dump(ccs, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CC_UFO_DS.pickle"), 'wb'))


# %%

cc_long = cc_long.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
cc_short = cc_short.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)


figure(figsize=(15,12))
outergs = GridSpec(2, 1, wspace=0.3, hspace=0.3, height_ratios=[0.3, 0.7])
gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[0], wspace=0.3, hspace=0.3)
ax = subplot(gs1[0])
plot(cc_long, alpha = 0.5, linewidth=1)
plot(cc_long.mean(1), linewidth = 4, color = 'red', label = "DS")
legend()
xlabel("ufo (s)")
axvline(0)
ax = subplot(gs1[1])
plot(cc_short, alpha = 0.5, linewidth=1)
plot(cc_short.mean(1), linewidth = 4, color = 'red', label = "DS")
legend()
xlabel("ufo (s)")
axvline(0)

# gs2 = GridSpecFromSubplotSpec(len(datasets)//4, 4, subplot_spec=outergs[1], wspace=0.3, hspace=0.3)
gs2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outergs[1], wspace=0.3, hspace=0.3)

for i, s in enumerate(perievent.keys()):
    gs3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs2[i//4, i%4], wspace=0.3, hspace=0.3)

    ax = subplot(gs3[1,0])
    plot(cc_short[s], linewidth=2)
    xlabel("ufo (s)")
    xlim(cc_short.index[0], cc_short.index[-1])
    axvline(0)

    subplot(gs3[0,0], sharex = ax)
    plot(perievent[s], '|', markersize=1)
    title(s.split("/")[-1])

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Cross-corr_UFO_DS.pdf"))
# show()

# %%
cc_long2 = pd.DataFrame.from_dict(cc_long2)
cc_short2 = pd.DataFrame.from_dict(cc_short2)

cc_long2 = cc_long2.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
cc_short2 = cc_short2.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)


figure(figsize=(15,12))
outergs = GridSpec(2, 1, wspace=0.3, hspace=0.3, height_ratios=[0.3, 0.7])
gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[0], wspace=0.3, hspace=0.3)
ax = subplot(gs1[0])
plot(cc_long2, alpha = 0.5, linewidth=1)
plot(cc_long2.mean(1), linewidth = 4, color = 'red', label = "UFO")
legend()
xlabel("DS (s)")
ax = subplot(gs1[1])
plot(cc_short2, alpha = 0.5, linewidth=1)
plot(cc_short2.mean(1), linewidth = 4, color = 'red', label = "UFO")
legend()
xlabel("DS (s)")

gs2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outergs[1], wspace=0.3, hspace=0.3)

for i, s in enumerate(perievent.keys()):
    gs3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs2[i//4, i%4], wspace=0.3, hspace=0.3)

    ax = subplot(gs3[1,0])
    plot(cc_short2[s], linewidth=2)
    xlabel("ufo (s)")
    xlim(cc_short2.index[0], cc_short2.index[-1])
    axvline(0)

    subplot(gs3[0,0], sharex = ax)
    plot(perievent2[s], '|', markersize=1)
    title(s.split("/")[-1])



tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Cross-corr_DS_UFO.pdf"))





# %%
# Spikes cross-corr

figure(figsize=(8,30))

gs = GridSpec(len(all_cc_ufo_dg), 2, wspace=0.3, hspace=0.5)

for i, s in enumerate(all_cc_ufo_dg.keys()):

    for j, state in enumerate(['wake', 'sws']):

        gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i, j], wspace=0.3, hspace=0.3)

        ax = subplot(gs2[0, 0])
        imshow(all_cc_ufo_dg[s][state].T, aspect='auto')
        title(f"{s.split('/')[-1]} - {state}")

        ax = subplot(gs2[1,0])
        plot(all_cc_ufo_dg[s][state], alpha=0.5)
        plot(all_cc_ufo_dg[s][state].mean(1), linewidth=2)
        xlabel("Time UFO (s)")
        axvline(0)
        # title(s.split("/")[-1])

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Cross-corr_UFO_DG_spikes.pdf"))