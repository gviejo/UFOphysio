import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import yaml
from matplotlib.gridspec import GridSpecFromSubplotSpec

from functions.functions import load_mean_waveforms
from ufo_detection import *
from matplotlib.pyplot import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from scipy.signal import correlate, correlation_lags

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

with open(os.path.join(data_directory, 'epochs_addg.yaml'), 'r') as f:
    epochs_addg = yaml.safe_load(f)



err_ahvs = {
    "ufo": [],
    "ds": []
    }
ufo_ahvs = {
    "dg": {},
    "adn": {}
}

def cross_corr(x, y):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    corr = correlate(x, y, mode='same', method='fft') / len(x)
    lags = correlation_lags(len(x), len(y), mode='same')

    return lags, corr

for s in datasets:
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
        sleep_ep = epochs[epochs.tags == "sleep"]
        sws_ep = nwb['sws']
        # rem_ep = nwb['rem']

        # Waveform classification
        meanwavef, maxch = load_mean_waveforms(os.path.join(path, "kilosort4"))
        spikes.maxch = np.hstack([chs for chs in maxch.values()])

        nwb.close()

    else:

        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sleep_ep = data.epochs['sleep']
        sws_ep = data.read_neuroscope_intervals('sws')
        #rem_ep = data.read_neuroscope_intervals('rem')
        # Waveform classification
        meanwavef, maxch = load_mean_waveforms(path)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])


    spikes.location = [v.lower() for v in spikes.location.values]
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)

    # Restricting ufos to period when no click happens
    events = np.array(epochs_addg[s])
    all_ep = nap.IntervalSet(start=np.sort(np.hstack([sleep_ep.start, wake_ep.start])), end = np.sort(np.hstack([sleep_ep.end, wake_ep.end])), metadata={"tags":events})
    sws_ep = sws_ep.intersect(all_ep[all_ep.tags == "Sleep"])
    ufo_ts_sws = ufo_ts.restrict(sws_ep)

    # Separate CA1 and DG neurons based on maxch
    location = spikes.location.copy()
    for i in location.index:
            if spikes.location[i] == "hpc":
                if spikes.maxch[i] < channels_separation[basename]:
                    location[i] = "ca1"
                else:
                    location[i] = "dg"
    spikes.location = location

    spikes = spikes[spikes.rate > 0.1]

    spikes = spikes[(spikes.location == 'adn') | (spikes.group == 0)]

    # Filtering out non hd adn
    grp_spikes = spikes[spikes.location == 'adn']
    thl_idx = grp_spikes.index
    tc2 = nap.compute_tuning_curves(
        grp_spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'],
        epochs=wake_ep
    )
    SI = nap.compute_mutual_information(tc2)['bits/spike']
    grp_spikes = grp_spikes[SI > 0.3]
    adn_idx = grp_spikes.index
    else_idx = np.setdiff1d(spikes.index, thl_idx)
    tokeep = np.sort(np.hstack([else_idx, adn_idx]))
    spikes = spikes[tokeep]

    adn_spikes = spikes[spikes.location == 'adn']
    dg_spikes = spikes[spikes.location == 'dg']

    if len(adn_spikes) > 3 and len(dg_spikes) > 3:

        # Tuning curves
        ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()
        new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support).drop_short_intervals(3)

        hd_tc = nap.compute_tuning_curves(
            adn_spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'], epochs=new_wake_ep,
            # return_pandas=True
        )
        ahv_tc = nap.compute_tuning_curves(
            dg_spikes, ahv, bins=20, range=(-np.pi, np.pi), feature_names=['ahv'], epochs=new_wake_ep#, return_pandas=True
        )

        decoded_hd, P_hd = nap.decode_bayes(
            hd_tc,
            data=adn_spikes,
            epochs=sws_ep,
            bin_size=0.01,
            sliding_window_size=5,
            uniform_prior=True
        )
        decoded_ahv_dg, P_ahv = nap.decode_bayes(
            ahv_tc,
            data=dg_spikes,
            epochs=sws_ep,
            bin_size=0.01,
            sliding_window_size=5,
            uniform_prior=True
        )

        decoded_ahv_adn = nap.Tsd(decoded_hd.t, np.unwrap(decoded_hd.d), time_support=sws_ep).derivative().smooth(0.1)

        ahv_dg_ufo = nap.compute_perievent(np.abs(decoded_ahv_dg), ufo_ts_sws, (-0.5, 0.5))
        ahv_adn_ufo = nap.compute_perievent(np.abs(decoded_ahv_adn), ufo_ts_sws, (-0.5, 0.5))

        ufo_ahvs["dg"][s] = np.nanmean(ahv_dg_ufo, 1).as_series()
        ufo_ahvs["adn"][s] = np.nanmean(ahv_adn_ufo, 1).as_series()

        # err = np.power(ahv_adn.values - ahv_dg.values, 2)  # noqa
        # err = np.abs(ahv_adn.values - ahv_dg.values)
        # err = nap.Tsd(t=ahv_adn.index.values, d=err, time_support=ahv_adn.time_support)

        # _, err = cross_corr(ahv_adn.restrict(sws_ep).values, ahv_dg.restrict(sws_ep).values)
        # err = nap.Tsd(t=ahv_adn.t, d=err, time_support=ahv_adn.time_support)

        # cc_ufo = nap.compute_perievent_continuous(err, ufo_ts.restrict(sws_ep), (-0.5, 0.5), ep=sws_ep)
        # cc_ufo = cc_ufo.as_dataframe()
        # cc_ds = nap.compute_perievent_continuous(err, ds_ts.restrict(sws_ep), (-0.5, 0.5), ep=sws_ep)
        # cc_ds = cc_ds.as_dataframe()
        #
        # err_ahvs["ufo"].append(cc_ufo.mean(1))
        # err_ahvs["ds"].append(cc_ds.mean(1))

        # cc_ufo = cc_ufo - cc_ufo.mean(0)
        # cc_ufo = cc_ufo / cc_ufo.std(0)
        # cc_ds = cc_ds - cc_ds.mean(0)
        # cc_ds = cc_ds / cc_ds.std(0)

###############################################################################
# PDF figure
###############################################################################
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import zscore

pdf_path = os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_AHV_decoding_ADN_DG.pdf")

labels = {"dg": "DG (decoded AHV)", "adn": "ADN (d/dt decoded HD)"}

with PdfPages(pdf_path) as pdf:

    # ── Page 1: peri-UFO |AHV|, DG vs ADN ──────────────────────────────────
    fig1, axes = subplots(1, 2, figsize=(10, 5), sharey=False)
    fig1.suptitle("Peri-UFO decoded |AHV| during SWS", fontsize=11)

    for ax, grp in zip(axes, ["dg", "adn"]):
        traces = []
        for s, ser in ufo_ahvs[grp].items():
            ax.plot(ser.index, ser.values, alpha=0.4, linewidth=0.8)
            traces.append(ser.values)
        if traces:
            min_len = min(len(t) for t in traces)
            mean_tr = np.nanmean([t[:min_len] for t in traces], axis=0)
            t_ax = list(ufo_ahvs[grp].values())[0].index[:min_len]
            ax.plot(t_ax, mean_tr, color='k', linewidth=1.8, label='mean')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(labels[grp], fontsize=9)
        ax.set_xlabel("Time from UFO (s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|Decoded AHV| (rad/s)")

    tight_layout()
    pdf.savefig(fig1)
    close(fig1)

    # ── Page 2: z-scored overlay — DG and ADN on same axes per session ──────
    sessions = sorted(set(ufo_ahvs["dg"].keys()) & set(ufo_ahvs["adn"].keys()))
    n_sess = len(sessions)
    if n_sess > 0:
        ncols = min(4, n_sess)
        nrows = int(np.ceil(n_sess / ncols))
        fig2, axes2 = subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        fig2.suptitle("Per-session peri-UFO |AHV|: DG (blue) vs ADN (orange)", fontsize=10)

        for idx, s in enumerate(sessions):
            ax = axes2[idx // ncols][idx % ncols]
            for grp, color in zip(["dg", "adn"], ["C0", "C1"]):
                if s in ufo_ahvs[grp]:
                    ser = ufo_ahvs[grp][s]
                    z = zscore(ser.values)
                    ax.plot(ser.index, z, color=color, linewidth=1.2, label=grp.upper())
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            ax.set_title(s.split("/")[-1], fontsize=7)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            if idx == 0:
                ax.legend(fontsize=6)

        # hide unused axes
        for idx in range(n_sess, nrows * ncols):
            axes2[idx // ncols][idx % ncols].set_visible(False)

        tight_layout()
        pdf.savefig(fig2)
        close(fig2)
