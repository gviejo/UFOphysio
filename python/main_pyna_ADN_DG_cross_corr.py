import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpecFromSubplotSpec

from functions.functions import load_mean_waveforms
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

cross_corrs = {}
tuning_curves = {}
mua_cross_corrs = {}


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
    spikes.maxch = np.hstack([chs for chs in maxch.values()])
    location = spikes.location.copy()
    location[(spikes.maxch < 30) & (spikes.location == "hpc")] = "ca1"
    location[(spikes.maxch >= 30) & (spikes.location == "hpc")] = "dg"
    spikes.location = location

    spikes = spikes[(spikes.location == 'adn') | (spikes.location == "dg") | (spikes.location == "ca1")]

    spikes = spikes[spikes.rate > 1.0]

    ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()

    new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)

    ###############################################################################################
    # CROSS-CORRELOGRAMS & TUNING CURVES
    ###############################################################################################

    tuning_curves[s] = {
        "hd" : nap.compute_tuning_curves(
            spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'], epochs=new_wake_ep
        ),
        "location": spikes.location,
        "maxch": spikes.maxch,
    }

    binsizes = {
        'wak': 0.05,
        'sws': 0.002,
    }
    windows = {
        'wak': 1,
        'sws': 0.5,
    }


    cross_corrs[s] = {}
    for loc in ["dg", "ca1"]:
        for state, ep in zip(["sws", "wak"], [sws_ep, new_wake_ep]):
            cc = nap.compute_crosscorrelogram(
                (
                    spikes[spikes.location == "adn"],
                    spikes[spikes.location == loc]
                ),
                binsizes[state],
                windows[state],
                ep, norm=True
            )
            cross_corrs[s][f'cc_{state}_{loc}'] = cc

    for state, ep in zip(["sws", "wak"], [sws_ep, new_wake_ep]):
        cc = nap.compute_crosscorrelogram(
            (
                spikes[spikes.location == "adn"],
                spikes[spikes.location.isin(["dg", "ca1"])]
            ),
            binsizes[state],
            windows[state],
            ep, norm=True
        )
        cross_corrs[s][f'cc_{state}'] = cc

    mua_cross_corrs[s] = {}
    for state, ep in zip(["sws", "wak"], [sws_ep, new_wake_ep]):
        cc = nap.compute_eventcorrelogram(
            spikes[spikes.group == 0],
            spikes[spikes.location == "adn"].to_tsd(),
            binsizes[state],
            windows[state],
            ep, norm=True
        )
        order = spikes[spikes.group == 0].maxch.sort_values().index
        mua_cross_corrs[s][f'cc_{state}'] = cc[order]


#################################################################################################
# Plot tuning curves
#################################################################################################

# %%
# All cross-corrs for each session
figure(figsize=(15, 100))
gs = GridSpec(len(cross_corrs), 1, wspace=0.3, hspace=0.3)

for k, s in enumerate(cross_corrs.keys()):

    gs2 = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[k,0], wspace=0.3, hspace=0.3)

    for i, e in enumerate(["wak", "sws"]):
        for j, loc in enumerate(["ca1", "dg"]):
            gs3 = GridSpecFromSubplotSpec(2,1, subplot_spec=gs2[j,i])
            ax = subplot(gs3[0, 0])
            cc = cross_corrs[s][f'cc_{e}_{loc}']
            order = cc.loc[-0.0:0.01].mean(0).sort_values().index[::-1]
            if len(cc.columns) > 0:
                plot(cc[order], alpha=0.5, linewidth=1)
                plot(cc[order].mean(1), color='k', linewidth=4)
                title(f"CC {e.upper()} - {loc.upper()} \n {s.split('/')[-1]}", fontsize=10)
                axvline(0, color='k', linestyle='--')
                ax = subplot(gs3[1, 0])
                imshow(cc[order].values.T, aspect='auto', cmap='jet', extent=(cc.index[0], cc.index[-1], 0, cc.shape[1]))

        # Sorted by max channel
        ax = subplot(gs2[2, i])
        cc = mua_cross_corrs[s][f'cc_{e}']
        # Zscoring
        cc = (cc - cc.mean(0)) / cc.std(0)
        imshow(cc.values.T, aspect='auto', cmap='jet', extent=(cc.index[0], cc.index[-1], 0, cc.shape[1]), origin="upper")
        title(f"MUA CC {e.upper()} - ADN \n {s.split('/')[-1]}", fontsize=10)
        axvline(0, color='k', linestyle='--')
        ylabel("Depth (maxch sorted)")


savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Cross_corr_ADN_HPC.pdf"))



# %%

figure(figsize=(30, 300))

# how many neurons per dataset
total_n = sum([cross_corrs[s]['cc_sws'].shape[1] for s in cross_corrs.keys()])
ncols = 4
nrows = total_n // ncols + int(total_n % ncols > 0) + 2* len(cross_corrs) + 1 # extra row between datasets

gs = GridSpec(nrows, ncols, wspace=0.3, hspace=0.3)

count = 0

colors = {"adn":"red", "ca1":"green", "dg":"#ff7f0e", "hpc":"#2ca02c"}

for i, s in enumerate(cross_corrs.keys()):
    n = cross_corrs[s]['cc_sws'].shape[1]
    tc = tuning_curves[s]['hd']

    for j in range(n):

        gs3 = GridSpecFromSubplotSpec(1,4, subplot_spec=gs[count//ncols, count%ncols], wspace=0.3)

        pair = cross_corrs[s]['cc_sws'].columns[j]

        for col, k in enumerate(pair):
            ax1 = subplot(gs3[0, col], projection='polar')
            tc.sel(unit=k).plot(ax=ax1, color=colors[tuning_curves[s]['location'].loc[k]])

        xticks([])
        yticks([])

        ax1 = subplot(gs3[0, 2])
        cc = cross_corrs[s]['cc_sws'][pair]
        plot(cc)
        xticks([])
        # yticks([])
        title(f"SWS CC {pair[0]}-{pair[1]}", fontsize=10)
        axvline(0, color='k', linestyle='--')

        ax2 = subplot(gs3[0, 3])
        cc = cross_corrs[s]['cc_wak'][pair]
        plot(cc)
        xticks([])
        # yticks([])
        title(f"Wake CC {pair[0]}-{pair[1]}", fontsize=10)
        axvline(0, color='k', linestyle='--')

        if j == 0:
            title(s.split('/')[-1], fontsize=16)

        count += 1

    count += 2 * ncols - (count % ncols)  # skip to next row

subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Cross_corr_ADN_DG.pdf"))
