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

nss_turn = {
    "wake": {
        "lmn": {"left": [], "right": []},
        "adn": {"left": [], "right": []}
    },
    "sws": {
        "lmn": {"left": [], "right": []},
        "adn": {"left": [], "right": []}
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

    if len(spikes) > 15:
        print(s)
        tuning_curves = tuning_curves.sel(unit=spikes.keys())
        spikes.peak = tuning_curves.idxmax(dim="0").values
        decoded, P = nap.decode_bayes(
            tuning_curves,
            data=spikes,
            epochs=sws_ep,
            bin_size=0.01,
            sliding_window_size=3,
            uniform_prior=True
        )

        # LMN UFOS
        ufo_ep_lmn, ufo_tsd_lmn = loadUFOs(path)
        nSS_lmn = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
        nSS_adn = nap.load_file(os.path.join(data.path, "nSS_ADN.npz"))

        ep = position[['x', 'z']].time_support.loc[[0]]
        bin_size = 0.04
        lin_velocity = computeLinearVelocity(position[['x', 'z']], ep, bin_size)
        lin_velocity = lin_velocity*100.0
        # ahv_wake = computeAngularVelocity(position['ry'], ep, bin_size)
        ahv_wake = np.unwrap(position['ry']).bin_average(bin_size, ep).derivative()

        ahv_sleep = np.unwrap(decoded).bin_average(bin_size, sws_ep).derivative()
        #
        window_size = {"wake": (-1.0, 1.0), "sws": (-0.5, 0.5)}

        peaks = {}
        bin_sizes = {"wake": 0.1, "sws": 0.01}
        window_sizes = {"wake": (-3, 3), "sws": (-0.5, 0.5)}

        for i, (ep, ep_name, ahv) in enumerate(zip([wake_ep, sws_ep], ["wake", "sws"], [ahv_wake, ahv_sleep])):

            # Search for transition
            thr = np.percentile(np.abs(ahv), 70)
            peaks[ep_name] = {
                "left": ahv[scipy.signal.find_peaks(ahv, height=thr)[0]],
                "right": ahv[scipy.signal.find_peaks(ahv*-1, height=thr)[0]]
            }

            for loc, nSS in zip(["lmn", "adn"], [nSS_lmn, nSS_adn]):

                for turn in ["left", "right"]:

                    tmp = nap.compute_perievent_continuous(nSS, peaks[ep_name][turn], window_size[ep_name], ep=ep)
                    tmp = tmp.as_dataframe()
                    tmp = tmp - tmp.mean(0)
                    tmp = tmp / tmp.std(0)
                    nss_turn[ep_name][loc][turn].append(tmp.mean(1))


for ep_name in nss_turn.keys():
    for loc in ["lmn", "adn"]:
        for turn in ["left", "right"]:
            nss_turn[ep_name][loc][turn] = pd.concat(nss_turn[ep_name][loc][turn], axis=1)

colors = {"left": "blue", "right": "red"}

figure(figsize=(18, 10))
gs = GridSpec(1, 2, wspace=0.3) # WAKE VS SWS
for i, ep_name in enumerate(nss_turn.keys()):
    gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, i], hspace=0.3) # LMN VS ADN

    for j, loc in enumerate(["adn", "lmn"]):
        subplot(gs2[j, 0])
        for turn in ["left", "right"]:
            plot(nss_turn[ep_name][loc][turn].mean(axis=1), '-', color=colors[turn], label=turn)
            fill_between(nss_turn[ep_name][loc][turn].index, nss_turn[ep_name][loc][turn].mean(axis=1) - nss_turn[ep_name][loc][turn].std(axis=1), nss_turn[ep_name][loc][turn].mean(axis=1) + nss_turn[ep_name][loc][turn].std(axis=1), alpha=0.3, color=colors[turn])
        legend()
        title(f"{ep_name.upper()} - {loc.upper()}")
        xlabel("Time from turn (s)")
        grid()

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_5.pdf"), dpi=100)


figure(figsize=(40, 120))
n_rows = nss_turn["wake"]["lmn"]["left"].shape[1]
gs = GridSpec(n_rows, 2, wspace=0.3) # WAKE VS SWS
for i, ep_name in enumerate(nss_turn.keys()):
    for k in range(n_rows):
        gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[k, i], hspace=0.3) # LMN VS ADN
        for j, loc in enumerate(["lmn", "adn"]):
            subplot(gs2[0, j])
            for turn in ["left", "right"]:
                tmp = nss_turn[ep_name][loc][turn].iloc[:, k]
                tmp = nap.Tsd(tmp).bin_average(0.0005).smooth(0.01).as_series()
                plot(tmp, '-', color=colors[turn], label=turn, linewidth=4)
                # fill_between(tmp.index, tmp - tmp.std(), tmp + tmp.std(), alpha=0.3, color=colors[turn])
            legend()
            title(f"{ep_name.upper()} - {loc.upper()}")
            xlabel("Time from turn (s)")
            grid()

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_6.pdf"), dpi=100)