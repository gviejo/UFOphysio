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

tuning_curves = {
}

maxchs = []

for s in datasets:
# for s in ['ADN-HPC/B5100/B5101/B5101-250502']:
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
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)

    # Waveform classification
    spikes.location = [v.lower() for v in spikes.location.values]
    meanwavef, maxch = load_mean_waveforms(path)
    maxchs.append(maxch)
    spikes.maxch = np.hstack([chs for chs in maxch.values()])
    location = spikes.location.copy()
    location[(spikes.maxch<30) & (spikes.location == "hpc")] = "ca1"
    location[(spikes.maxch>=30) & (spikes.location == "hpc")] = "dg"
    spikes.location = location

    spikes = spikes[(spikes.location == 'adn') | (spikes.group == 0)]

    spikes = spikes[spikes.rate > 1.0]

    ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()

    new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)


    ###############################################################################################
    # TUNING CURVES
    ###############################################################################################

    tuning_curves[s] = {
        "hd" : nap.compute_tuning_curves(
            spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'], epochs=new_wake_ep
        ),
        "ahv" : nap.compute_tuning_curves(
            spikes, ahv, bins=20, range=(-np.pi, np.pi), feature_names=['ahv'], epochs=new_wake_ep
        ),
        "pf" : nap.compute_tuning_curves(
            spikes, position.loc[['x', 'z']], bins=(12, 12), feature_names=['x', 'z']
        ),
        "location": spikes.location,
        "cc" : nap.compute_eventcorrelogram(
            spikes, ufo_ts, 0.002, 0.1, sws_ep, norm=True
        )
    }



#################################################################################################
# Plot tuning curves
#################################################################################################
figure(figsize=(15, 200))

# how many neurons per dataset
total_n = sum([tuning_curves[s]['hd'].shape[0] for s in datasets])
ncols = 3
nrows = total_n // ncols + int(total_n % ncols > 0) + 2* len(datasets) + 1 # extra row between datasets

gs = GridSpec(nrows, ncols, wspace=0.3, hspace=0.3)

count = 0

colors = {"adn":"red", "ca1":"green", "dg":"#ff7f0e", "hpc":"#2ca02c"}

for i, s in enumerate(datasets):
    n = tuning_curves[s]['hd'].shape[0]

    for j in range(n):

        gs3 = GridSpecFromSubplotSpec(2,3, subplot_spec=gs[count//ncols, count%ncols], wspace=0.3)

        color = colors[tuning_curves[s]['location'].values[j]]

        ax1 = subplot(gs3[0,0:2], projection='polar')
        tc = tuning_curves[s]['hd'][j]
        tc.plot(ax=ax1, color=color)
        xticks([])
        yticks([])

        ax1 = subplot(gs3[1,0])
        tc = tuning_curves[s]['ahv'][j]
        tc.plot(ax=ax1, color=color)
        xticks([])
        # yticks([])
        title("")


        ax2 = subplot(gs3[1,1])
        tc = tuning_curves[s]['pf'][j]
        tc.plot(ax=ax2, cmap='viridis')
        ax2.set_aspect("equal")
        xticks([])
        yticks([])
        title("")

        ax3 = subplot(gs3[:,2])
        cc = tuning_curves[s]['cc'].iloc[:,j]
        plot(cc, color=color)
        axvline(0, color='k', linestyle='--')

        if j == 0:
            title(s.split('/')[-1], fontsize=16)

        count += 1

    count += 2 * ncols - (count % ncols)  # skip to next row

subplots_adjust(top=0.98, bottom=0.02)

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Tuning_curves_B51_ADDG.pdf"))
