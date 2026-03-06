"""
PAired ufos ahv analysis
"""

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
from functions.functions import computeLinearVelocity, computeAngularVelocity
# import pynacollada as pyna
from ufo_detection import *
import warnings

warnings.filterwarnings("ignore")

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

datasets = np.genfromtxt(os.path.join(data_directory, 'datasets_LMN_ADN.list'), delimiter='\n', dtype=str, comments='#')

# LMN ufo channels
ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter=' ', dtype=str, comments='#')
ufo_channels = {a[0]: a[1:].astype('int') for a in ufo_channels}

# ADN ufo channels
ufo_channels_ADN = np.genfromtxt(os.path.join(data_directory, 'channels_UFO_ADN.txt'), delimiter=' ', dtype=str, comments='#')
ufo_channels_ADN = {a[0]: a[1:].astype('int') for a in ufo_channels_ADN}

for s in ["LMN-ADN/A5044/A5044-240401A"]:

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

    spikes = spikes[(spikes.location == "adn") | (spikes.location == "lmn")]
    spikes = spikes.getby_threshold("rate", 1)
    tuning_curves = nap.compute_tuning_curves(
        spikes, position['ry'], 60, range=(0, 2 * np.pi), epochs=position.time_support
    )
    SI = nap.compute_mutual_information(tuning_curves)
    spikes.set_info(SI=SI["bits/spike"])
    spikes = spikes[spikes.SI > 0.1]

    tuning_curves = tuning_curves.sel(unit=spikes.keys())
    spikes.peak = tuning_curves.idxmax(dim="0").values
    decoded, P = nap.decode_bayes(
        tuning_curves,
        data=spikes,
        epochs=sws_ep,
        bin_size=0.005,
        sliding_window_size=5,
        uniform_prior=True
    )

    # LMN UFOS
    ufo_ep_lmn, ufo_tsd_lmn = loadUFOs(path)
    nSS_lmn = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
    ufo_tsd_lmn = ufo_tsd_lmn.restrict(nSS_lmn.time_support)
    # nSS_lmn = nSS_lmn.bin_average(0.0005)
    # ufo_tsd_lmn = ufo_tsd_lmn.value_from(nSS_lmn)

    # ADN UFOS
    ufo_tsd_adn = nap.load_file(os.path.join(path, data.basename + '_ufo_tsd_ADN.npz'))
    nSS_adn = nap.load_file(os.path.join(data.path, "nSS_ADN.npz"))
    ufo_tsd_adn = ufo_tsd_adn.restrict(nSS_adn.time_support)
    # nSS_adn = nSS_adn.bin_average(0.0005)

    # # Taking only event above 4 std
    # ufo_tsd_adn = ufo_tsd_adn[ufo_tsd_adn > 4]
    # ufo_tsd_lmn = ufo_tsd_lmn[ufo_tsd_lmn > 4]

    # Making categories
    categories = {}
    window = 0.01

    tmp_adn = nap.Tsd(t=ufo_tsd_adn.t, d=np.arange(len(ufo_tsd_adn)), time_support=ufo_tsd_adn.time_support)
    tmp_lmn = nap.Tsd(t=ufo_tsd_lmn.t, d=np.arange(len(ufo_tsd_lmn)), time_support=ufo_tsd_lmn.time_support)

    diff = tmp_adn.t[ufo_tsd_lmn.value_from(tmp_adn)] - ufo_tsd_lmn.t

    # LMN only
    categories[0] = ufo_tsd_lmn[np.abs(diff) > window].value_from(nSS_lmn) # LMN events not paired with ADN

    # LMN followed by ADN
    categories[1] = ufo_tsd_lmn[(diff>0)&(diff<window)].value_from(nSS_lmn) # LMN events paired with ADN

    diff = tmp_lmn.t[ufo_tsd_adn.value_from(tmp_lmn)] - ufo_tsd_adn.t

    # ADN only
    categories[2] = ufo_tsd_adn[np.abs(diff) > window].value_from(nSS_adn) # ADN events not paired with LMN

    # ADN followed by LMN
    categories[3] = ufo_tsd_adn[(diff>0)&(diff<window)].value_from(nSS_adn) # ADN events paired with LMN


    categories = nap.TsGroup(categories,
                             metadata={"name": ["LMN", "LMN->ADN", "ADN", "ADN->LMN"]}
                             )

    categories = categories.restrict(sws_ep)

    # Plotting LFP for each category for 10 examples
    data2 = nap.EphysReader(path, format='neurosuite')
    lfp = data2[s.split("/")[-1] + ".dat"]

    adn_group = ufo_channels_ADN[s][0]
    lmn_group = ufo_channels[s][0]

    figure(figsize=(25, 90))
    gs = GridSpec(4,1, hspace=0.5)
    for i, cat_name in enumerate(categories.name.values):
        ts = categories[i].t[np.argsort(categories[i])[-12:]]
        gs2 = GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[i], wspace=0.1)
        for j, t in enumerate(ts):
            subplot(gs2[j//4, j%4])
            tmp = lfp.get(t-0.02, t+0.02)
            tmp = nap.apply_bandpass_filter(tmp, (600, 3000), 20000)
            offset = 0
            for group, color in zip([lmn_group, adn_group], ["blue", "red"]):
                for k, ch in enumerate(tmp.group[tmp.group == group].index):
                    plot(tmp[:, ch] + offset, color=color, alpha=1)
                    offset += 1000
                offset += 1000
            axvline(t, color="black", ls="--", alpha=0.5)
            ylim(-1000, offset)
            if j % 4 == 0 and j // 4 == 0:
                title(cat_name)

    savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_7_examples.pdf"), dpi=100)


    nSS = np.hstack((nSS_lmn.values[:,None], nSS_adn.values[:,None]))
    nSS = nap.TsdFrame(nSS_lmn.t, nSS, columns = ["lmn", "adn"])

    figure()
    markers = ['o', 's', '^', 'd']
    markercols = ['blue', 'red', 'cyan', 'magenta']
    for i, cat_name in enumerate(categories.name.values):
        tsd = categories[i].value_from(nSS)

        plot(np.log2(tsd['lmn']), np.log2(tsd['adn']), markers[i], color=markercols[i], label=cat_name, alpha=1, markersize=1)
    axvline(np.log2(3), color='k', ls='--')
    axhline(np.log2(3), color='k', ls='--')
    xlabel("LMN nSS (log2)")
    ylabel("ADN nSS (log2)")
    legend()
    tight_layout()
    title(s.split("/")[-1])
    savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_8_examples.pdf"), dpi=100)





    print(categories)
    ep = position.time_support[0]
    bin_size = 0.02
    ahv_wake = np.unwrap(position['ry']).bin_average(bin_size, ep).derivative()
    ahv_sleep = np.unwrap(decoded).bin_average(bin_size, sws_ep).derivative()
    #
    # window_size = {"wake": (-1.0, 1.0), "sws": (-0.5, 0.5)}
    #
    # peaks = {}
    #
    # for i, (ep, ep_name, ahv) in enumerate(zip([wake_ep, sws_ep], ["wake", "sws"], [ahv_wake, ahv_sleep])):
    #     ahv_corr = nap.compute_event_trigger_average(categories, ahv, bin_size, window_size[ep_name], ep)
    #     abs_ahv_corr = nap.compute_event_trigger_average(categories, np.abs(ahv), bin_size, window_size[ep_name], ep)
    #     for j, cat_name in enumerate(categories.name.values):
    #         ahvs[ep_name][cat_name].append(ahv_corr.loc[j].as_series())
    #         abs_ahvs[ep_name][cat_name].append(abs_ahv_corr.loc[j].as_series())
    #
    #     # Search for transition
    #     thr = np.percentile(np.abs(ahv), 70)
    #     peaks[ep_name] = {
    #         "left": ahv[scipy.signal.find_peaks(ahv, height=thr)[0]],
    #         "right": ahv[scipy.signal.find_peaks(ahv*-1, height=thr)[0]]
    #     }
    #     for turn in ["left", "right"]:
    #         bin_sizes = {"wake": 0.1, "sws": 0.01}
    #         window_sizes = {"wake": 3, "sws": 0.5}
    #         tmp = nap.compute_eventcorrelogram(categories, peaks[ep_name][turn], bin_sizes[ep_name], window_sizes[ep_name], ep=ep)
    #         tmp.columns = ["LMN->ADN", "ADN"]
    #         ccs_turns[ep_name][turn]["LMN->ADN"].append(tmp["LMN->ADN"])
    #         ccs_turns[ep_name][turn]["ADN"].append(tmp["ADN"])
    #



