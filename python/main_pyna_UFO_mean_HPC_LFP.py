# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-10-23 11:58:20
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-10-23 13:42:53


import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions import *
from ufo_detection import *

from matplotlib.pyplot import *

# nap.nap_config.set_backend("jax")

from functions.functions import loadXML


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


perilfp = {}

for s in datasets:
# for s in ["ADN-HPC/B5100/B5102/B5102-250915"]:
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
    # sws_ep = data.read_neuroscope_intervals('sws')
    #rem_ep = data.read_neuroscope_intervals('rem')

    ufo_ep, ufo_ts = loadUFOs(path)


    if ufo_ep is None:
        print("No UFOs detected in this session {}".format(s))
    else:
        ufo_ts = ufo_ts.restrict(sleep_ep)

        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel
        filename = data.basename + ".eeg"

        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels, 1250)


        lfp = nap.TsdFrame(t=timestep, d=fp)

        # load XML info
        num_channels, fs, shank_to_channel, shank_to_keep = loadXML(path)

        mean_lfp = []
        ch = channels[0][shank_to_keep[0]]

        for c in ch:
            tmp1 = lfp[:,c]
            # tmp1 = tmp1 - np.mean(tmp1)
            # tmp1 = tmp1 / np.std(tmp1)
            tmp = nap.compute_perievent(tmp1, ufo_ts, window=(-1, 1))
            mean_lfp.append(np.nanmean(tmp, 1).d)

        mean_lfp = np.array(mean_lfp)
        mean_lfp = pd.DataFrame(mean_lfp.T, index=tmp.t, columns = ch)

        # perilfp = nap.compute_perievent_continuous(lfp, ufo_ts, minmax=(-1, 1))

        perilfp[s] = mean_lfp

###############################################################################################
# PLOTTING
###############################################################################################
import matplotlib.pyplot as plt
import matplotlib as mpl

n_sessions = len(perilfp)
ncols = 4
nrows = int(np.ceil(n_sessions / ncols))

fig = plt.figure(figsize=(ncols * 5, nrows * 6))
gs = GridSpec(nrows, ncols, hspace=0.5, wspace=0.45)

for i, s in enumerate(perilfp.keys()):
    mean_lfp = perilfp[s]
    T = mean_lfp.index.values
    n_ch = mean_lfp.shape[1]

    gs2 = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[i // ncols, i % ncols],
        height_ratios=[1, 1], hspace=0.25
    )

    # --- waterfall ---
    ax0 = fig.add_subplot(gs2[0, 0])
    offset = np.abs(mean_lfp.values).max() * 1.2
    colors = mpl.colormaps["viridis"](np.linspace(0, 1, n_ch))
    for j in range(n_ch):
        ax0.plot(T, mean_lfp.values[:, j] - j * offset,
                 color=colors[j], lw=0.8)
    ax0.axvline(0, color="k", lw=1, ls="--", alpha=0.7)
    ax0.set_xlim(T[0], T[-1])
    ax0.set_yticks([])
    ax0.set_ylabel("Channels", fontsize=7)
    ax0.set_title(s.split("/")[-1], fontsize=8, fontweight="bold")
    ax0.tick_params(labelbottom=False, bottom=False)
    ax0.spines[["top", "right", "left"]].set_visible(False)

    # --- heatmap ---
    ax1 = fig.add_subplot(gs2[1, 0])
    vmax = np.abs(mean_lfp.values).max()
    im = ax1.imshow(
        mean_lfp.values.T,
        aspect="auto",
        extent=[T[0], T[-1], n_ch - 0.5, -0.5],
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax1.axvline(0, color="k", lw=1, ls="--", alpha=0.7)
    ax1.set_xlabel("Time from UFO (s)", fontsize=7)
    ax1.set_ylabel("Channel", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.spines[["top", "right"]].set_visible(False)

    # shared colorbar
    cb = plt.colorbar(im, ax=ax1, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("LFP (µV)", fontsize=6)

fig.suptitle("Mean HPC LFP around UFOs", fontsize=11, fontweight="bold", y=1.01)

plt.savefig(
    os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_mean_HPC_LFP.pdf"),
    bbox_inches="tight",
    dpi=150,
)
plt.savefig(
    os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_mean_HPC_LFP.png"),
    bbox_inches="tight",
    dpi=150,
)

