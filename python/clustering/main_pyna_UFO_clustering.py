# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-07-13 16:08:51
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-12-01 18:21:24

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

import sys
sys.path.append("..")

from ufo_detection import get_memory_map, loadUFOs
from functions.functions import *

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

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


# for s in datasets:
for s in ["LMN-ADN/A5043/A5043-230301A"]:
# for s in ["LMN-ADN/A5043/A5043-230306A"]:
# for s in ["LMN-ADN/A5044/A5044-240401A"]:
# for s in ["LMN-ADN/A6002/A6002-210408A"]:
# for s in ["LMN/A1414/A1414-200929A"]:

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

    # spikes = spikes[spikes.location == "adn"]
    # spikes = spikes[(spikes.location=="adn") | (spikes.location=="lmn")]

    if ufo_ts is not None:

        # Load LFP data
        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel
        # sign_channels = channels[ufo_channels[s][0]]        
        sign_channels = channels[1]
        # sign_channels = np.hstack((channels[1], channels[2]))
        filename = data.basename + ".dat"
        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

        lfp = nap.TsdFrame(t=timestep, d=fp)

        # UFOs during wake
        ufo_wake = ufo_ts.restrict(position.time_support)
        # ufo_wake = ufo_ts.restrict(sws_ep)

        segs = np.zeros((len(ufo_wake), len(sign_channels), 201))

        for i, t in enumerate(ufo_wake.t):
            segment = lfp.get(t-0.01, t+0.01)[:, sign_channels]
            fseg = nap.apply_highpass_filter(segment, 600, fs=20000)
            fseg = fseg.get(t-0.005, t+0.005)
            fseg = fseg * np.blackman(fseg.shape[0])[:, np.newaxis]
            fseg = fseg - fseg.mean(1).values[:,None]
            segs[i,:,:] = fseg.values.T

        segs_2d = segs.reshape(segs.shape[0], -1)

        freqs = np.fft.fftfreq(segs.shape[-1], d=1/20000)
        F = np.fft.fft(segs, axis=-1)
        F = F[:, :, freqs >= 600]

        F_2d = np.abs(F).reshape(F.shape[0], -1)

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        from scipy.ndimage import gaussian_filter
        from sklearn.preprocessing import StandardScaler
        import xarray as xr
        import hdbscan as hdbscan

        n_clusters = 10
        pca = PCA(n_components=20)
        F_2d = StandardScaler().fit_transform(F_2d)
        X = pca.fit_transform(F_2d)

        labels = KMeans(n_clusters=n_clusters).fit_predict(X)

        # labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method='leaf').fit_predict(X)

        Y = TSNE(n_components=2, perplexity=5).fit_transform(X)

        figure()
        scatter(Y[:,0], Y[:,1], s=20, c=labels, cmap='tab10')


        figure()
        for i, k in enumerate(np.unique(labels)):
            subplot(2, len(np.unique(labels)), i+1)
            meanseg = np.mean(segs[labels==k,:,:], 0)
            for j in range(meanseg.shape[0]):
                plot(np.linspace(-3,3,meanseg.shape[1]), meanseg[j,:] + j*500)
            title("Cluster {}".format(k))
            subplot(2, len(np.unique(labels)), i+1+len(np.unique(labels)))
            imshow(meanseg, aspect='auto', extent=[-3,3,0, meanseg.shape[0]], cmap='jet')


        figure()
        plot(position['x'], position['z'], linewidth=0.5, alpha=0.5)
        tmp = ufo_wake.value_from(position[['x', 'z', 'ry']])
        color = hsv_to_rgb( np.vstack(( (tmp['ry'].values % (2*np.pi)) / (2*np.pi), np.ones(len(tmp))*0.8, np.ones(len(tmp))*0.8 )).T )
        scatter(tmp['x'], tmp['z'], 100, color=color)



        subufos = nap.TsGroup({
            k: ufo_wake[labels==k] for k in np.unique(labels)
        }, time_support = position.time_support)

        tcufos = nap.compute_tuning_curves(subufos, position['ry'], 60, range=(0, 2*np.pi), epochs=position.time_support, return_pandas=True)
        pfufos = nap.compute_tuning_curves(subufos, position[['x', 'z']], 30, epochs = position.time_support.loc[[0]])

        tcufos2 = smoothAngularTuningCurves_old(tcufos, window = 20, deviation = 2.0)
        pfufos = xr.apply_ufunc(
            gaussian_filter,
            pfufos.fillna(0),
            kwargs={"sigma": (0, 2, 2)},
            input_core_dims=[["unit", "x", "z"]],
            output_core_dims=[["unit", "x", "z"]],
        )

        figure()
        gs = GridSpec(3, len(tcufos.columns), height_ratios=[1, 1, 1])
        for i, k in enumerate(tcufos.columns):
            ax = subplot(gs[0, i], projection='polar')
            plot(tcufos[k])
            plot(tcufos2[k], color='r')
            title("Cluster {}".format(k))
            ax = subplot(gs[1, i])
            imshow(pfufos.sel(unit=k).values.T, aspect='equal', origin='lower', cmap='jet')

            subplot(gs[2, i], aspect='equal')
            plot(position['x'], position['z'], linewidth=0.5, alpha=0.5)
            
            tmp = ufo_wake[labels==k].value_from(position[['x', 'z', 'ry']])
            color = hsv_to_rgb( np.vstack(( tmp['ry'].values / (2*np.pi), np.ones(len(tmp)), np.ones(len(tmp)) )).T )
            scatter(tmp['x'], tmp['z'], 50, color=color)

        show()




        # offset_tc = {}
        # offset_si = []
        # offset_si_all = []
        # ry = position['ry']
        # offset_values = np.linspace(-2, 2, 100)
        # offset_si_all = []
        # offset_tc_all = []

        # for offset in offset_values:
        #     new_ry = nap.Tsd(t=ry.t-offset, d=ry.values, 
        #         time_support=ry.time_support)
        #     # plot(new_ry, color=cm.viridis((offset - offset_values.min()) / (offset_values.max() - offset_values.min())))

        #     tc = nap.compute_tuning_curves(subufos, new_ry, 16, range=(0, 2*np.pi), epochs=new_ry.time_support, return_pandas=True)
        #     tc2 = smoothAngularTuningCurves_old(tc, window = 20, deviation = 2.0)
        #     offset_tc[offset] = tc2
        #     offset_si.append(nap.compute_mutual_information(
        #         nap.compute_tuning_curves(subufos, new_ry, 16, range=(0, 2*np.pi), epochs=new_ry.time_support)
        #         )['bits/spike'].values)

        #     # control 
        #     tc = nap.compute_tuning_curves(nap.TsGroup({0: ufo_wake}), new_ry, 16, range=(0, 2*np.pi), epochs=new_ry.time_support)
        #     offset_si_all.append(nap.compute_mutual_information(tc)['bits/spike'].values[0])
        #     offset_tc_all.append(tc.values[0])

        # offset_si = np.array(offset_si)
        # offset_si = pd.DataFrame(index=offset_values, data=offset_si, columns=tcufos.columns)

        # offset_si_all = pd.Series(offset_si_all, index=offset_values)
        
        # # offset_si = offset_si - offset_si.mean(0)
        # # offset_si = offset_si / offset_si.std(0)

        # # offset_si_all = offset_si_all - offset_si_all.mean()
        # # offset_si_all = offset_si_all / offset_si_all.std()

        # offset_tc_all = pd.DataFrame(np.array(offset_tc_all), index=offset_values)

        # figure()
        # plot(offset_si.mean(1))
        # plot(offset_si_all)



        # figure()
        # gs = GridSpec(1, len(tcufos.columns))
        # for i, k in enumerate(tcufos.columns):
            
        #     ax = subplot(gs[0, i])
            
        #     tmp = np.array([offset_tc[offset][k].values for offset in offset_tc.keys()])
        #     imshow(tmp, aspect='auto', extent=[0, 2*np.pi, -2, 2], origin='lower', cmap='jet')

        # show()
        
        

