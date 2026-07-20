import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpecFromSubplotSpec

from python.functions.functions import load_mean_waveforms
from ufo_detection import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans

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

channels_separation = np.genfromtxt(os.path.join(data_directory, 'channels_CA1_DG.txt'), delimiter =' ', dtype = str, comments ='#')
channels_separation = {a[0].split("/")[-1] : a[1].astype('int') for a in channels_separation}

tuning_curves = {
}

maxchs = []
ch_pos = {}

for s in datasets:
# for s in ['ADN-HPC/B5100/B5101/B5101-250502']:
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
        # rem_ep = nwb['rem']

        # Waveform classification
        meanwavef, maxch = load_mean_waveforms(os.path.join(path, "kilosort4"))
        maxchs.append(maxch)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])

        nwb.close()

    else:

        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        #rem_ep = data.read_neuroscope_intervals('rem')
        # Waveform classification
        meanwavef, maxch = load_mean_waveforms(path)
        maxchs.append(maxch)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])


    spikes.location = [v.lower() for v in spikes.location.values]
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)

    # Separate CA1 and DG neurons based on maxch
    location = spikes.location.copy()
    for i in location.index:
            if spikes.location[i] == "hpc":
                if spikes.maxch[i] < channels_separation[basename]:
                    location[i] = "ca1"
                else:
                    location[i] = "dg"
    spikes.location = location


    # maxch_values = spikes.maxch[location == "hpc"].values.reshape(-1, 1)
    # idx = np.where(location == "hpc")[0]
    # if len(maxch_values) > 0:
    #     clu = KMeans(n_clusters=2, random_state=0).fit(maxch_values)
    #     map_ = {np.argmin(clu.cluster_centers_.flatten()): "ca1", np.argmax(clu.cluster_centers_.flatten()): "dg"}
    #     for i, label in enumerate(clu.labels_):
    #         location[idx[i]] = map_[label]
    #     # location[(spikes.maxch<30) & (spikes.location == "hpc")] = "ca1"
    #     # location[(spikes.maxch>=30) & (spikes.location == "hpc")] = "dg"
    #     spikes.location = location
    # groups = spikes.metadata[['location', 'maxch']].groupby("location").groups
    # ch_pos[s] = {loc: spikes.maxch[groups[loc]].values for loc in groups.keys()}

    ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()

    new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)

    tmp = {}
    for g in meanwavef.keys():
        for i in range(meanwavef[g].shape[0]):
            n = spikes.index[spikes.group == g][i]
            tmp[n] = meanwavef[g][i]

    spikes = spikes[(spikes.location == 'adn') | (spikes.group == 0)]

    spikes = spikes[spikes.rate > 1.0]

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
        "maxch": spikes.maxch,
        "meanwavef": tmp,
        "group": spikes.group,
        "cc" : nap.compute_eventcorrelogram(
            spikes, ufo_ts, 0.002, 0.1, sws_ep, norm=True
        )
    }



#################################################################################################
# Plot tuning curves — one session per page
#################################################################################################
from matplotlib.backends.backend_pdf import PdfPages

colors = {"adn":"red", "ca1":"green", "dg":"#ff7f0e", "hpc":"#2ca02c"}
ncols = 3

pdf_path = os.path.expanduser("~/Dropbox/UFOPhysio/figures/Tuning_curves_B51_ADDG.pdf")

with PdfPages(pdf_path) as pdf:
    for s in datasets:
        print("Plotting session {}".format(s))


        order = ["ca1", "dg", "adn"]
        neuron_ids = []
        for group in order:
            neuron_ids += tuning_curves[s]['location'][tuning_curves[s]['location'] == group].index.values.tolist()

        # # Sort by maxch & group
        # # neuron_ids = tuning_curves[s]['maxch'].sort_values().index.values
        # neuron_ids = []
        # for group in np.unique(tuning_curves[s]['group']):
        #     neuron_ids += tuning_curves[s]['maxch'][tuning_curves[s]['group'] == group].sort_values().index.values.tolist()
        # neuron_ids = np.array(neuron_ids)

        n = len(neuron_ids)
        if n == 0:
            continue

        nrows = n // ncols + int(n % ncols > 0)
        fig = figure(figsize=(40, nrows * 5))
        fig.suptitle(s.split('/')[-1], fontsize=16, y=1.01)

        gs = GridSpec(nrows, ncols, wspace=0.3, hspace=0.5, figure=fig)

        for j, nid in enumerate(neuron_ids):
            gs3 = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[j // ncols, j % ncols], wspace=0.3)

            color = colors[tuning_curves[s]['location'][nid]]


            ax0 = subplot(gs3[0, 0])
            meanwavef = tuning_curves[s]['meanwavef'][nid]
            imshow(meanwavef.T, aspect='auto', cmap='viridis')
            if tuning_curves[s]['location'][nid] in ['ca1', 'dg']:
                axhline(channels_separation[s.split('/')[-1]], color='w', linestyle='--')
            xticks([])
            yticks([0, meanwavef.shape[1]//2, meanwavef.shape[1]-1], [0, meanwavef.shape[1]//2, meanwavef.shape[1]])
            title("group: {}, location: {}".format(tuning_curves[s]['group'][nid], tuning_curves[s]['location'][nid]))

            ax1 = subplot(gs3[0, 1], projection='polar')
            tc = tuning_curves[s]['hd'].sel(unit=nid)
            tc.plot(ax=ax1, color=color)
            xticks([])
            yticks([])

            ax2 = subplot(gs3[1, 0])
            tc = tuning_curves[s]['ahv'].sel(unit=nid)
            tc.plot(ax=ax2, color=color)
            xticks([])
            title("")

            ax3 = subplot(gs3[1, 1])
            tc = tuning_curves[s]['pf'].sel(unit=nid)
            tc.plot(ax=ax3, cmap='viridis')
            ax3.set_aspect("equal")
            xticks([])
            yticks([])


            ax4 = subplot(gs3[:, 2])
            cc = tuning_curves[s]['cc'][nid]
            plot(cc, color=color)
            axvline(0, color='k', linestyle='--')
            xlabel("Ufo")

        subplots_adjust(top=0.95, bottom=0.02)
        pdf.savefig(fig, bbox_inches='tight')
        close(fig)

    # # Plot maxch distribution
    # n = len(ch_pos)
    # ncols = 4
    # nrows = n // ncols + int(n % ncols