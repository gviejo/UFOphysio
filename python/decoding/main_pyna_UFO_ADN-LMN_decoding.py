# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-07-13 16:08:51
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-12-03 11:32:22

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os

from matplotlib import pyplot as plt
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations

from sklearn.decomposition import KernelPCA
from matplotlib.colors import hsv_to_rgb
sys.path.append("..")
from functions import *
from ufo_detection import *
from scipy import signal
import functools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

mdpec = {}

for s in datasets:
# for s in ["LMN-ADN/A5043/A5043-230301A"]:

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
    rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)

    nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
    ufo_ts = ufo_ts.value_from(nSS)
    ufo_ep = ufo_ep[ufo_ts > 5]
    ufo_ts = ufo_ts[ufo_ts > 5]

    spikes = spikes[spikes.location == "adn"]
    # spikes = spikes[(spikes.location=="adn") | (spikes.location=="lmn")]

    if ufo_ts is not None:
        tuning_curves = nap.compute_tuning_curves(
            spikes, position['ry'], 60, range=(0, 2 * np.pi), epochs=position.time_support
        )

        SI = nap.compute_mutual_information(tuning_curves)

        spikes.set_info(SI=SI["bits/spike"])


        if len(spikes)>8:

            # tuning_curves = tuning_curves.sel(unit=spikes.keys())
            spikes.peak = tuning_curves.idxmax(dim="0").values
            spikes.order = np.argsort(np.argsort(spikes.peak.values))

            # spikes = spikes[spikes.SI > 0.3]

            ep = sws_ep

            decoded, P = nap.decode_bayes(
                tuning_curves,
                data=spikes,
                epochs = ep,
                bin_size=0.005,
                sliding_window_size=5,
                uniform_prior=True
            )

            # # Ring decoding
            # count = spikes.count(0.02, ep).smooth(0.1)
            # X = count.values
            # X = X - X.mean(axis=0)
            # X = X / X.std(axis=0)
            # Xt = Xt[Xt.sum(1) > np.percentile(Xt.sum(1), 0.5)]
            # imap = KernelPCA(n_components=2, kernel='cosine')
            # Xt = imap.fit_transform(X)
            #
            decoded2 = nap.Tsd(t=decoded.t, d=np.unwrap(decoded.values), time_support=decoded.time_support)
            angspeed = decoded2.derivative().abs().smooth(0.02)
            dpec = nap.compute_perievent_continuous(angspeed, ufo_ts.restrict(ep), (-0.3, 0.3), ep=ep)

            mdpec[s] = np.nanmean(dpec, axis=1)

            # # Raster
            # figure()
            # cmap = plt.get_cmap("hsv")
            # ep = sws_ep
            # for i, (n,o,p) in enumerate(zip(spikes.index, spikes.order, spikes.peak)):
            #     neuron_spikes = spikes[n].restrict(ep).t
            #     h = (p % (2 * np.pi)) / (2 * np.pi)
            #     plot(neuron_spikes, np.ones(len(neuron_spikes)) * o, '|', ms =40,
            #          color = hsv_to_rgb([h,1,1])
            #          )
            # [axvline(t, color='r', lw=1) for t in ufo_ts[(spikes.to_tsd().count(ep=ufo_ep)>4).values].restrict(ep).t]
            # # show()

            # from sklearn.manifold import TSNE, MDS, Isomap
            # from sklearn.decomposition import PCA
            # from sklearn.cluster import KMeans, DBSCAN
            # from sklearn.preprocessing import StandardScaler
            # import umap

            # X = spikes.count(ep=ufo_ep).values
            # X = X[X.sum(1) > 1]
            # # tsne = TSNE(n_components=2, perplexity=20, random_state=0)
            # # Y = tsne.fit_transform(X)

            # Y = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

            # #
            # # Y = PCA(n_components=2).fit_transform(X)

            # labels = KMeans(n_clusters=20, random_state=42).fit_predict(Y)
            # # labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(X)

            # # X_mds = MDS(n_components=2).fit_transform(X)

            # # Y = Isomap(n_components=2, n_neighbors=20).fit_transform(X)
            # X2 = StandardScaler().fit_transform(X)[np.argsort(labels)]

            # plt.figure(figsize=(8, 6))
            # plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=50)
            # plt.xlabel("Dimension 1")
            # plt.ylabel("Dimension 2")
            # plt.show()



mdpec2 = pd.concat([mdpec[s].as_series() for s in mdpec.keys()], axis=1)



figure(figsize = (8, 10))
subplot(211)
plot(mdpec)
xlabel("Time from UFO (s)")
ylabel("Angular speed (rad/s)")
axvline(0)
subplot(212)
plot(mdpec.mean(1), '-k', lw=2)
fill_between(mdpec.index, mdpec.mean(1)-mdpec.std(1), mdpec.mean(1)+mdpec.std(1), color='k', alpha=0.2)
xlabel("Time from UFO (s)")
ylabel("Angular speed (rad/s)")
axvline(0)
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_ADN-LMN_decoding_angspeed.png"), dpi=300)



# fig = figure()
# ax = fig.add_subplot(projection='3d')
# for i in range(pec.shape[1]):
#     ax.plot(np.cos(pec[:,i]), np.sin(pec[:,i]), np.ones(pec.shape[0])*i, '-', alpha=0.2, linewidth=2)
#
# show()
#
#
# figure()
# ax = subplot(211)
# plot(spikes.to_tsd("peak").restrict(ep), '|k', ms=10)
# [axvline(t, color='r', lw=1) for t in ufo_ts.restrict(ep).t]
#
#
# ax2 = subplot(212, sharex=ax)
# imshow(P.values.T, aspect='auto', origin='lower', extent=[ep.start[0], ep.end[0], 0, 2 * np.pi], cmap='jet')
# [axvline(t, color='r', lw=2) for t in ufo_ts.restrict(ep).t]
# show()
