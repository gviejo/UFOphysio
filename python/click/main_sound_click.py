# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-10 12:39:17

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations

sys.path.append("../")
from ufo_detection import *
from functions import *
from matplotlib.pyplot import *
import yaml

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

alldatasets = yaml.safe_load(open(os.path.join(data_directory,'datasets_SOUND.yaml'), 'r'))['struct']


ccs = {
    "wak": {
        "adn": {},
        "lmn": {},
        "ds":  {}
    },
    "sws": {
        "adn": {},
        "lmn": {},
        "ds": {}
    }
}

peths = {
    "adn": {},
    "lmn": {},
    "ds": {}
}

for struct in alldatasets.keys():
# for struct in ["adn"]:
    print(f"{struct}")

    datasets = alldatasets[struct]["click_10s"]
    if "click_5s" in alldatasets[struct].keys():
        datasets += alldatasets[struct]["click_5s"]

    for ss in datasets:
        s = ss[0]
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

        ufo_ep, ufo_ts = loadUFOs(path)

        # LMN default
        n_channels = 2
        epoch_int = 0

        if struct == "adn":
            ds_ep, ds_ts = loadDentateSpikes(path)

            epoch_int = ss[1]
            n_channels = ss[2]

        analogin, ts = get_memory_map(os.path.join(data.path, data.basename+"_"+str(epoch_int)+"_analogin.dat"), n_channels, frequency=20000)
        peaks,_ = scipy.signal.find_peaks(np.diff(analogin[:,0]), height=2000)
        peaks+=1
        ttl = nap.Ts(t=ts[peaks])

        # Adding intervals if epoch_int is not 0
        if epoch_int != 0:
            epochs = pd.read_csv(os.path.join(data.path, "Epoch_TS.csv"), index_col=False, header=None).values
            start = np.sum(np.diff(epochs, 1)[0:epoch_int])
            ttl = nap.Ts(t = ttl.t + start)

        assert len(ttl) > 0, "No sound TTL found!"
        print(np.mean(np.diff(ttl.t)), np.std(np.diff(ttl.t)))

        if struct == "adn":
            assert np.std(np.diff(ttl.t)) < 0.1, "TTLs are not regular!"

        ttl.save(data.path + "/ttl_sound")


        ###############################################################################################
        group = {0:ufo_ts}
        if struct == "adn":
            group[1] = ds_ts
        group = nap.TsGroup(group)

        # CCG
        for e, ep in zip(['wak', 'sws'], [wake_ep, sleep_ep]):
            cc = nap.compute_eventcorrelogram(
                group,
                ttl, 0.001, 0.2, ep=ep, norm=True
            )

            ccs[e][struct][s] = cc[0] #/len(ttl.restrict(ep))

            if struct == "adn":
                ccs[e]['ds'][s] = cc[1]#/len(ttl.restrict(ep))

        # PETH
        peth = nap.compute_perievent(
            group,
            ttl.restrict(sws_ep), 0.1)

        peths[struct][s] = peth[0]
        if struct == "adn":
            peths["ds"][s] = peth[1]

        # Writing for neuroscope
        peaks2 = ttl.as_units('ms').index.values
        datatowrite = peaks2
        n = len(peaks2)
        texttowrite = np.repeat(np.array(['SOUND 1']), n)
        evt_file = os.path.join(data.path, data.basename+'.evt.py.snd')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()



for e in ccs.keys():
    for struct in ccs[e].keys():
        ccs[e][struct] = pd.DataFrame.from_dict(ccs[e][struct])


figure(figsize=(12,6))
gs = GridSpec(2,3)
for i, struct in enumerate(['lmn', 'adn', 'ds']):
    e = "sws"
    subplot(gs[0,i])
    count = 0
    for j ,s in enumerate(peths[struct].keys()):
        gr = peths[struct][s]
        plot(gr.to_tsd(np.arange(0, len(gr))+count+1), 'o', markersize=1)
        count += len(gr)

    ylabel("Trial")
    axvline(0)
    xlim(-0.1, 0.1)
    title(struct)

    subplot(gs[1,i])
    plot(ccs[e][struct], alpha=0.3, linewidth=1)
    plot(ccs[e][struct].mean(1), color='k', linewidth=1)
    axvline(0)
    xlim(-0.1, 0.1)
    if i == 0:
        ylabel("UFO / Sound CCG")
tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Cross-corr_sound.pdf"))



###############################################################################################
# Saving
datatosave = {
    "ccs":ccs,
    "peths":{struct:{s:peths[struct][s].to_tsd().as_series() for s in peths[struct].keys()} for struct in peths.keys()}
}

import _pickle as cPickle
cPickle.dump(datatosave, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/cc_sound.pickle"), 'wb'))
