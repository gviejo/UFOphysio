# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-05 17:16:59

import pynapple as nap
from matplotlib import rcParams, pyplot as plt
from matplotlib.pyplot import *

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from ufo_detection import *

from functions.functions import centerTuningCurves, smoothAngularTuningCurves, circmean

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

datasets = {
    'adn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    'lmn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    'psb':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')
}

ccs_long = {r:{e:[] for e in ['wak', 'rem', 'sws']} for r in ['adn', 'lmn', 'psb']}
ccs_short = {r:{e:[] for e in ['wak', 'rem', 'sws']} for r in ['adn', 'lmn', 'psb']}

SI_thr = {
    'adn':0.2, 
    'lmn':0.1,
    'psb':0.0
    }

for r in datasets.keys():
    for s in datasets[r]:

        print(s)

        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, s)
        basename = os.path.basename(path)
        filepath = os.path.join(path, "kilosort4", basename + ".nwb")

        if os.path.exists(filepath):
            nwb = nap.load_file(filepath)
            spikes = nwb['units']

            position = []
            columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            for k in columns:
                position.append(nwb[k].values)
            position = np.array(position)
            position = np.transpose(position)
            position = nap.TsdFrame(
                t=nwb['x'].t,
                d=position,
                columns=columns,
                time_support=nwb['position_time_support'])

            epochs = nwb['epochs']
            wake_ep = epochs[epochs.tags == "wake"]
            sws_ep = nwb['sws']
            rem_ep = nwb['rem']

            ufo_ep, ufo_ts = loadUFOs(path)

            spikes = spikes[spikes.location == r]

            if ufo_ts is not None:

                tuning_curves = nap.compute_tuning_curves(
                    spikes,
                    position['ry'],
                    bins=60,
                    range=(0, 2 * np.pi),
                    epochs=position.time_support[0],
                    feature_names=['angle']
                )

                tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

                SI = nap.compute_mutual_information(tuning_curves)['bits/spike']

                spikes.SI = SI

                spikes = spikes[spikes.SI>SI_thr[r]]

                names = [s.split("/")[-1] + "_" + str(n) for n in spikes.keys()]

                for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
                    cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.01, 0.6, ep, norm=True)
                    cc.columns = names
                    ccs_long[r][e].append(cc)

                    cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.001, 0.015, ep, norm=True)
                    cc.columns = names
                    ccs_short[r][e].append(cc)


            else:
                print("No ufo in "+s)

for r in ccs_long.keys():
    for e in ccs_long[r].keys():
        ccs_long[r][e] = pd.concat(ccs_long[r][e], axis=1)
        ccs_short[r][e] = pd.concat(ccs_short[r][e], axis=1)


datatosave = {"ccs_long":ccs_long, "ccs_short":ccs_short}

import _pickle as cPickle
cPickle.dump(datatosave, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CC_UFO_PSB.pickle"), 'wb'))



rcParams.update({'font.size': 15})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
lw = 2

figure(figsize = (10, 12))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

gs = GridSpec(4,3)
for i, r in enumerate(['lmn', 'adn', 'psb']):
    for j, e in enumerate(ccs_long[r].keys()):
        subplot(gs[i,j])
        if i == 0:
            title(e)
        if j == 0:
            ylabel(r, rotation=0, labelpad = 30)
        tmp = ccs_long[r][e].values
        tmp = tmp - tmp.mean(0)
        tmp = tmp / tmp.std(0)
        imshow(tmp.T, aspect='auto', cmap = 'jet')
        x = ccs_long[r][e].index.values
        xticks([0, len(x)//2, len(x)], [x[0], 0.0, x[-1]])

for j, e in enumerate(ccs_long[r].keys()):
    subplot(gs[-1,j])
    for i, r in enumerate(['lmn', 'adn', 'psb']):
        plot(ccs_long[r][e].mean(1), color = colors[i], linewidth=lw, label=r)
    axvline(0.0)
    xlim(x[0], x[-1])
    legend()
tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/ALL_CC_UFO_Long.png"))


figure(figsize = (10, 12))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

gs = GridSpec(4,3)
for i, r in enumerate(['lmn', 'adn', 'psb']):
    for j, e in enumerate(ccs_short[r].keys()):
        subplot(gs[i,j])
        if i == 0:
            title(e)
        if j == 0:
            ylabel(r, rotation=0, labelpad = 30)
        tmp = ccs_short[r][e].values
        tmp = tmp - tmp.mean(0)
        tmp = tmp / tmp.std(0)        
        tmp = tmp[:,np.where(~np.isnan(np.sum(tmp, 0)))[0]]
        imshow(tmp.T, aspect='auto', cmap = 'jet')
        x = ccs_short[r][e].index.values
        xticks([0, len(x)//2, len(x)], [x[0], 0.0, x[-1]])

for j, e in enumerate(ccs_short[r].keys()):
    subplot(gs[-1,j])
    for i, r in enumerate(['lmn', 'adn', 'psb']):
        plot(ccs_short[r][e].mean(1), color = colors[i], linewidth=lw, label=r)
    axvline(0.0)
    xlim(x[0], x[-1])
    legend()

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/ALL_CC_UFO_Short.png"))

