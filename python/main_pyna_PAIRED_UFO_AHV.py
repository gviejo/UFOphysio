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

ahvs = {
    "wake": {
        "LMN->ADN": [],
        # "LMN": [],
        "ADN": []
    },
    "sws": {
        "LMN->ADN": [],
        # "LMN": [],
        "ADN": []
    }
}

abs_ahvs = {
    "wake": {
        "LMN->ADN": [],
        # "LMN": [],
        "ADN": []
    },
    "sws": {
        "LMN->ADN": [],
        # "LMN": [],
        "ADN": []
    }
}

ccs_turns = {
    "wake": {
        "left": {
            "LMN->ADN": [],
            "ADN": []
        },
        "right": {
            "LMN->ADN": [],
            "ADN": []
        }
    },
    "sws": {
        "left": {
            "LMN->ADN": [],
            "ADN": [],
        },
        "right": {
            "LMN->ADN": [],
            "ADN": [],
        }
    }
}

for s in datasets:

    ###############################################################################################
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')

    spikes = spikes[(spikes.location == "adn") | (spikes.location == "lmn")]
    spikes = spikes.getby_threshold("rate", 1)
    tuning_curves = nap.compute_tuning_curves(
        spikes, position['ry'], 60, range=(0, 2 * np.pi), epochs=position.time_support
    )
    SI = nap.compute_mutual_information(tuning_curves)
    spikes.set_info(SI=SI["bits/spike"])
    spikes = spikes[spikes.SI > 0.1]

    if len(spikes) > 20:
        print(s)
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
        ufo_tsd_lmn = ufo_tsd_lmn.value_from(nSS_lmn)

        # ADN UFOS
        ufo_tsd_adn = nap.load_file(os.path.join(path, data.basename + '_ufo_tsd_ADN.npz'))

        # Taking only event above 4 std
        ufo_tsd_adn = ufo_tsd_adn[ufo_tsd_adn > 4]
        ufo_tsd_lmn = ufo_tsd_lmn[ufo_tsd_lmn > 4]

        # Making categories
        categories = {} # ADN events not paired with LMN

        window = 0.01
        ep = nap.IntervalSet(ufo_tsd_adn.t - window, ufo_tsd_adn.t)
        count = ufo_tsd_lmn.count(ep=ep)
        # categories[0] = nap.Ts(count[count > 0].t + window/2) # ADN events paired with LMN
        categories[0] = ufo_tsd_lmn
        categories[1] = nap.Ts(count[count == 0].t - window/2) # ADN events not paired with LMN
        # categories[2] = ufo_tsd_adn[ufo_tsd_adn.in_interval(ep) == 0]
        categories = nap.TsGroup(categories, metadata={"name": ["LMN->ADN", "ADN"]})

        if categories.rate.min() > 0.5:

            print(s, categories)

            #
            ep = position[['x', 'z']].time_support.loc[[0]]
            bin_size = 0.02
            # lin_velocity = computeLinearVelocity(position[['x', 'z']], ep, bin_size)
            # lin_velocity = lin_velocity*100.0
            # ahv_wake = computeAngularVelocity(position['ry'], ep, bin_size)
            ahv_wake = np.unwrap(position['ry']).bin_average(bin_size, ep).derivative()

            ahv_sleep = np.unwrap(decoded).bin_average(bin_size, sws_ep).derivative()
            #
            window_size = {"wake": (-1.0, 1.0), "sws": (-0.5, 0.5)}

            peaks = {}

            for i, (ep, ep_name, ahv) in enumerate(zip([wake_ep, sws_ep], ["wake", "sws"], [ahv_wake, ahv_sleep])):
                ahv_corr = nap.compute_event_trigger_average(categories, ahv, bin_size, window_size[ep_name], ep)
                abs_ahv_corr = nap.compute_event_trigger_average(categories, np.abs(ahv), bin_size, window_size[ep_name], ep)
                for j, cat_name in enumerate(categories.name.values):
                    ahvs[ep_name][cat_name].append(ahv_corr.loc[j].as_series())
                    abs_ahvs[ep_name][cat_name].append(abs_ahv_corr.loc[j].as_series())

                # Search for transition
                thr = np.percentile(np.abs(ahv), 70)
                peaks[ep_name] = {
                    "left": ahv[scipy.signal.find_peaks(ahv, height=thr)[0]],
                    "right": ahv[scipy.signal.find_peaks(ahv*-1, height=thr)[0]]
                }
                for turn in ["left", "right"]:
                    bin_sizes = {"wake": 0.1, "sws": 0.01}
                    window_sizes = {"wake": 3, "sws": 0.5}
                    tmp = nap.compute_eventcorrelogram(categories, peaks[ep_name][turn], bin_sizes[ep_name], window_sizes[ep_name], ep=ep)
                    tmp.columns = ["LMN->ADN", "ADN"]
                    ccs_turns[ep_name][turn]["LMN->ADN"].append(tmp["LMN->ADN"])
                    ccs_turns[ep_name][turn]["ADN"].append(tmp["ADN"])




for ep_name in ahvs.keys():
    # for cat_name in ahvs[ep_name].keys():
    for cat_name in ["LMN->ADN", "ADN"]:
        ahvs[ep_name][cat_name] = pd.concat(ahvs[ep_name][cat_name], axis=1)
        abs_ahvs[ep_name][cat_name] = pd.concat(abs_ahvs[ep_name][cat_name], axis=1)


colors = {"wake": "blue", "sws": "orange"}

# Plotting
# AHV perievent

figure(figsize=(18, 10))
gs = GridSpec(1, 2, hspace=0.3, wspace=0.3)
for i, ep_name in enumerate(ahvs.keys()):
    gs2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0])
    for j, cat_name in enumerate(ahvs[ep_name].keys()):
        subplot(gs2[i, j])
        plot(ahvs[ep_name][cat_name].mean(axis=1), '-', color=colors[ep_name])
        fill_between(ahvs[ep_name][cat_name].index, ahvs[ep_name][cat_name].mean(axis=1) - ahvs[ep_name][cat_name].std(axis=1), ahvs[ep_name][cat_name].mean(axis=1) + ahvs[ep_name][cat_name].std(axis=1), alpha=0.3)
        title(f"{ep_name} - {cat_name}")
        ylabel("AHV (rad/s)")
        grid()

for i, ep_name in enumerate(abs_ahvs.keys()):
    gs2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1])
    for j, cat_name in enumerate(abs_ahvs[ep_name].keys()):
        subplot(gs2[i, j])
        plot(abs_ahvs[ep_name][cat_name].mean(axis=1), '-', color=colors[ep_name])
        fill_between(abs_ahvs[ep_name][cat_name].index, abs_ahvs[ep_name][cat_name].mean(axis=1) - abs_ahvs[ep_name][cat_name].std(axis=1), abs_ahvs[ep_name][cat_name].mean(axis=1) + abs_ahvs[ep_name][cat_name].std(axis=1), alpha=0.3)
        title(f"{ep_name} - {cat_name}")
        ylabel("|AHV| (rad/s)")
        grid()

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_2.pdf"), dpi=300)


for ep_name in ccs_turns.keys():
    for turn in ccs_turns[ep_name].keys():
        for cat_name in ccs_turns[ep_name][turn].keys():
            ccs_turns[ep_name][turn][cat_name] = pd.concat(ccs_turns[ep_name][turn][cat_name], axis=1)


colors = {"left": "green", "right": "red"}

figure(figsize=(18, 10))
gs = GridSpec(1, 2, wspace=0.3) # WAKE VS SWS
for i, ep_name in enumerate(ccs_turns.keys()):
    gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, i]) # LMN VS ADN
    for j, cat_name in enumerate(ccs_turns[ep_name]["left"].keys()):
        subplot(gs2[0, j])
        for turn in ["left", "right"]:
            plot(ccs_turns[ep_name][turn][cat_name].mean(axis=1), '-', color=colors[turn], label=turn)
            # fill_between(ccs_turns[ep_name][turn][cat_name].index, ccs_turns[ep_name][turn][cat_name].mean(axis=1) - ccs_turns[ep_name][turn][cat_name].std(axis=1), ccs_turns[ep_name][turn][cat_name].mean(axis=1) + ccs_turns[ep_name][turn][cat_name].std(axis=1), alpha=0.3, color=colors[ep_name])
        legend()
        title(f"{ep_name} - {cat_name}")
        xlabel("Time from turn (s)")
        grid()

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_3.pdf"), dpi=300)

figure(figsize=(40, 120))
gs = GridSpec(1, 2, wspace=0.3) # WAKE VS SWS

n_rows = ccs_turns["wake"]["left"]["LMN->ADN"].shape[1]

for i, ep_name in enumerate(ccs_turns.keys()):
    gs2 = GridSpecFromSubplotSpec(n_rows, 2, subplot_spec=gs[0, i]) # LMN VS ADN
    for j in range(n_rows):
        for k, cat_name in enumerate(["LMN->ADN", "ADN"]):
            subplot(gs2[j, k])
            for turn in ["left", "right"]:
                plot(ccs_turns[ep_name][turn][cat_name].iloc[:,j], '-', color=colors[turn], label=turn)
                # fill_between(ccs_turns[ep_name][turn][cat_name].index, ccs_turns[ep_name][turn][cat_name].mean(axis=1) - ccs_turns[ep_name][turn][cat_name].std(axis=1), ccs_turns[ep_name][turn][cat_name].mean(axis=1) + ccs_turns[ep_name][turn][cat_name].std(axis=1), alpha=0.3, color=colors[ep_name])
            legend()
            title(f"{ep_name} - {cat_name}")
            xlabel("Time from turn (s)")
            grid()

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_4.pdf"), dpi=300)