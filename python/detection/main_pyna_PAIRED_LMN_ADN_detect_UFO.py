"""
This is to check if LMN ufo detection is equal to ADN ufo detection.
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

rates = {"wake": [], "sws": []}

pwr = {ep_name:
    {
        "LMN->ADN":
            {
                "adn": [],
                "lmn": []
            },
        "LMN":
            {
                "adn": [],
                "lmn": []
            },
        "ADN":
            {
                "adn": [],
                "lmn": []
            }
    }
    for ep_name in rates.keys()
}

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
    sws_ep = data.read_neuroscope_intervals('sws')

    # LMN UFOS
    ufo_ep_lmn, ufo_tsd_lmn = loadUFOs(path)
    nSS_lmn = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))

    if s not in ufo_channels_ADN.keys():
        print("No UFO channels specified for this session {}".format(s))
        break

    # try:
    #     ufo_tsd_adn = nap.load_file(os.path.join(path, data.basename + '_ufo_tsd_ADN.npz'))
    #     nSS_adn = nap.load_file(os.path.join(data.path, "nSS_ADN.npz"))
    # except:
    ###############################################################################################
    # MEMORY MAP
    ###############################################################################################
    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    sign_channels = channels[ufo_channels_ADN[s][0]]
    ctrl_channels = channels[ufo_channels_ADN[s][1]]
    filename = data.basename + ".dat"

    fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

    ufo_ep, ufo_tsd, nSS = detect_ufos_v2(fp, sign_channels, ctrl_channels, timestep)


    ufo_tsd.save(os.path.join(path, data.basename + '_ufo_tsd_ADN'))
    nSS.save(os.path.join(path, "nSS_ADN.npz"))

    nSS_adn = nSS
    ufo_tsd_adn = ufo_tsd
    ####################


    # Taking only event above 4 std
    ufo_tsd_adn = ufo_tsd_adn[ufo_tsd_adn > 4]

    # Making categories
    categories = {"LMN->ADN": [], # Paired events between ADN and LMN
                  "LMN": [] , # LMN events not paired with ADN
                  "ADN": []} # ADN events not paired with LMN

    window = 0.005
    ep = nap.IntervalSet(ufo_tsd_lmn.t, ufo_tsd_lmn.t + window)
    count = ufo_tsd_adn.count(ep=ep)
    categories["LMN->ADN"] = nap.Ts(count[count > 0].t - window/2)
    categories["LMN"] = nap.Ts(count[count == 0].t - window/2)
    categories["ADN"] = ufo_tsd_adn[ufo_tsd_adn.in_interval(ep) == 0]

    for i, (ep, ep_name) in enumerate(zip([wake_ep, sws_ep], ["wake", "sws"])):
        for cat_name in categories.keys():
            # lmn
            tmp = nap.compute_perievent_continuous(nSS_lmn,  categories[cat_name].restrict(ep), (-0.01, 0.01))
            pwr[ep_name][cat_name]["lmn"].append(tmp.as_dataframe())
            # adn
            tmp = nap.compute_perievent_continuous(nSS_adn,  categories[cat_name].restrict(ep), (-0.01, 0.01))
            pwr[ep_name][cat_name]["adn"].append(tmp.as_dataframe())


    # Compute perievent for each category and epoch
    for i, (ep, ep_name) in enumerate(zip([wake_ep, sws_ep], ["wake", "sws"])):
        psth_lmn_adn = nap.compute_perievent(ufo_tsd_adn.restrict(ep), ufo_tsd_lmn.restrict(ep), (-0.005, 0.01))
        psth_adn_lmn = nap.compute_perievent(ufo_tsd_lmn.restrict(ep), ufo_tsd_adn.restrict(ep), (-0.01, 0.005))

        rates[ep_name].append(pd.DataFrame([
                        np.sum(psth_lmn_adn.rate>0)/len(psth_lmn_adn),
                        np.sum(psth_adn_lmn.rate>0)/len(psth_adn_lmn)
                        ], index=["LMN->ADN", "LMN<-ADN"], columns=[s.split("/")[-1]]
                    )
        )

    # Value from for each type of event
    pwr2 = nap.TsdFrame(
        t=ufo_tsd_lmn.t,
        d=np.vstack((ufo_tsd_lmn.value_from(nSS_lmn).values, ufo_tsd_lmn.value_from(nSS_adn).values)).T,
        columns=["lmn", "adn"]
    )
    pwr3 = nap.TsdFrame(
        t=ufo_tsd_adn.t,
        d=np.vstack((ufo_tsd_adn.value_from(nSS_lmn).values, ufo_tsd_adn.value_from(nSS_adn).values)).T,
        columns=["lmn", "adn"]
    )

for k in rates.keys():
    rates[k] = pd.concat(rates[k], axis=1).T


for ep_name in pwr.keys():
    for cat_name in pwr[ep_name].keys():
        for region in pwr[ep_name][cat_name].keys():
            tmp = []
            for df in pwr[ep_name][cat_name][region]:
                tmp.append(df.mean(axis=1))
            pwr[ep_name][cat_name][region] = pd.concat(tmp, axis=1)

figure()
for i, ep_name in enumerate(rates.keys()):
    subplot(1, 2, i+1)
    plot(rates[ep_name].T, 'o-')
    title(ep_name)
    ylim(0, 1)
tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_0.pdf"), dpi=300)

colors = {"wake": "blue", "sws": "orange"}

figure(figsize=(15, 10))
gs = GridSpec(1, 2)
for i, ep_name in enumerate(pwr.keys()):
    gs2 = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, i])
    for j, cat_name in enumerate(pwr[ep_name].keys()):
        for k, region in enumerate(pwr[ep_name][cat_name].keys()):
            subplot(gs2[k, j])
            plot(pwr[ep_name][cat_name][region].mean(axis=1), '-', color=colors[ep_name])
            fill_between(pwr[ep_name][cat_name][region].index, pwr[ep_name][cat_name][region].mean(axis=1) - pwr[ep_name][cat_name][region].std(axis=1), pwr[ep_name][cat_name][region].mean(axis=1) + pwr[ep_name][cat_name][region].std(axis=1), alpha=0.3)
            title(f"{ep_name} - {cat_name} - {region}")
            ylim(-5, 15)
            if j == 0:
                ylabel(region)

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Pairing_UFO_ADN-LMN_1.pdf"), dpi=300)

# show()