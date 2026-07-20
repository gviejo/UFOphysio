import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import umap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import zscore
from sklearn.cluster import KMeans

from functions.functions import load_mean_waveforms
from ufo_detection import *
from reactivation import compute_reactivation
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

channels_separation = np.genfromtxt(os.path.join(data_directory, 'channels_CA1_DG.txt'), delimiter =' ', dtype = str, comments ='#')
channels_separation = {a[0].split("/")[-1] : a[1].astype('int') for a in channels_separation}

groups = ['adn', 'dg', 'ca1', 'hpc', 'all']

maxchs = []
ch_pos = {}

ufo_reacs = {g: {} for g in groups}
ufo_reacs_ctrl = {g: {} for g in groups}
ufo_reacs_ev = {g: {} for g in groups}  # per-eigenvector perievent, [g][s] = DataFrame (time x n_evecs)
evecs_store = {g: {} for g in groups}   # significant eigenvectors, [g][s] = array (n_neurons x n_evecs)
tc_store = {g: {} for g in groups}      # HD tuning curves, [g][s] = DataFrame (bins x neurons)
ahv_tc_store = {g: {} for g in groups}  # AHV tuning curves, [g][s] = DataFrame (bins x neurons)
weighted_tc_store = {g: {} for g in groups}      # weighted-avg HD TC per evec, [g][s] = list of arrays (n_bins,)
weighted_ahv_tc_store = {g: {} for g in groups}  # weighted-avg AHV TC per evec, [g][s] = list of arrays (n_bins,)
all_cc_ufo_reac = {}

for s in [  "ADN-HPC/B5100/B5102/B5102-250919",
            "ADN-HPC/B5100/B5107/B5107-260224",
            "ADN-HPC/B5100/B5107/B5107-260227",
            "ADN-HPC/B5100/B5107/B5107-260219",
          ]:
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
        maxchs.append(maxch)
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
        maxchs.append(maxch)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])


    spikes.location = [v.lower() for v in spikes.location.values]
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)

    # Restrict ufos to pre sleep and post sleep
    ufo_ts_pre = ufo_ts.restrict(sws_ep.intersect(sleep_ep[0]))
    ufo_ts_post = ufo_ts.restrict(sws_ep.intersect(sleep_ep[-1]))

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

    # Filtering out non hd adn
    grp_spikes = spikes[spikes.location == 'adn']
    thl_idx = grp_spikes.index
    tc2 = nap.compute_tuning_curves(
        grp_spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'],
        epochs=wake_ep
    )
    SI = nap.compute_mutual_information(tc2)['bits/spike']
    grp_spikes = grp_spikes[SI > 0.1]
    adn_idx = grp_spikes.index
    else_idx = np.setdiff1d(spikes.index, thl_idx)
    tokeep = np.sort(np.hstack([else_idx, adn_idx]))
    spikes = spikes[tokeep]

    # PANDas tuning curves
    ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()
    new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support).drop_short_intervals(3)
    tuning_curves = nap.compute_tuning_curves(
        spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'], epochs=new_wake_ep,
        return_pandas=True
    )
    ahv_tc = nap.compute_tuning_curves(
        spikes, ahv, bins=20, range=(-np.pi, np.pi), feature_names=['ahv'], epochs=new_wake_ep, return_pandas=True
    )


    if np.sum(spikes.location == "dg") > 2:

        bin_size = 0.01

        for g in groups:
            if g == 'all':
                grp_spikes = spikes
            elif g == 'hpc':
                grp_spikes = spikes[spikes.location.isin(['dg', 'ca1'])]
            else:
                grp_spikes = spikes[spikes.location == g]


            if len(grp_spikes) < 2:
                continue

            tc_store[g][s] = tuning_curves if g == 'all' else tuning_curves[grp_spikes.index]
            ahv_tc_store[g][s] = ahv_tc if g == 'all' else ahv_tc[grp_spikes.index]

            ##################################################
            # Compute reactivation
            ##################################################
            significant_evecs, significant_evals = compute_reactivation(grp_spikes, new_wake_ep, bin_size)
            evecs_store[g][s] = significant_evecs

            # Probability of participating = evec**2 (unit eigenvectors already sum to 1 when squared)
            wtcs, wahv_tcs = [], []
            for i in range(significant_evecs.shape[1]):
                probs = significant_evecs[:, i] ** 2
                # probs = significant_evecs[:,i]
                probs = probs / np.sum(probs)  # Normalize to sum to 1
                wtcs.append(tc_store[g][s].values @ probs)
                wahv_tcs.append(ahv_tc_store[g][s].values @ probs)
            weighted_tc_store[g][s] = wtcs
            weighted_ahv_tc_store[g][s] = wahv_tcs

            sleep_count = grp_spikes.count(bin_size, sws_ep)
            sleep_count = sleep_count.smooth(3*bin_size)
            sleep_count = sleep_count - sleep_count.mean(0)
            sleep_count = sleep_count / sleep_count.std(0)

            # Compute per-eigenvector reactivation scores, controlled by column-shuffle
            n_shuffles = 10
            ev_scores = []
            ev_scores_ctrl = []
            for i in range(significant_evecs.shape[1]):
                w = significant_evecs[:, i]
                P = np.outer(w, w)
                np.fill_diagonal(P, 0)

                score = np.einsum('ti,ij,tj->t', sleep_count.d, P, sleep_count.d)
                # score = score / np.abs(significant_evals[i])  # Normalize by eigenvalue to account for variance explained
                ev_scores.append(score)

                shuf_scores = []
                for _ in range(n_shuffles):
                    sleep_shuf = sleep_count.d.copy()
                    sleep_shuf = sleep_shuf[:, np.random.permutation(sleep_shuf.shape[1])]
                    score_shuf = np.einsum('ti,ij,tj->t', sleep_shuf, P, sleep_shuf)
                    # score_shuf = score_shuf / np.abs(significant_evals[i])
                    shuf_scores.append(score_shuf)
                ev_scores_ctrl.append(np.mean(shuf_scores, 0))

            ev_scores = np.array(ev_scores).T  # (n_timepoints, n_evecs)
            ev_scores_ctrl = np.array(ev_scores_ctrl).T  # (n_timepoints, n_evecs)

            # ev_scores = ev_scores - ev_scores_ctrl  # Subtract shuffle control to isolate structured reactivation

            # Total reactivation (sum across eigenvectors)
            reactivation = nap.Tsd(
                t=sleep_count.t,
                d=np.sum(ev_scores, 1) - np.sum(ev_scores_ctrl, 1),
                time_support=sleep_count.time_support
            )
            ufo_reacs[g][s] = {}
            for ep, ufo_ts_sws in zip(["pre", "post"], [ufo_ts_pre, ufo_ts_post]):
                ufo_reac = nap.compute_perievent(reactivation, ufo_ts_sws, window=(-0.5, 0.5))
                tmp = pd.DataFrame(np.nanmean(ufo_reac, 1).as_series())
                tmp.columns = [s.split("/")[-1]]
                ufo_reacs[g][s][ep] = tmp







###############################################################################
# Helpers
###############################################################################
def draw_session_page(fig, s, gs_reac, gs_bot, n_max):
    """Draw all groups for one session into pre-built reactivation and bottom gridspecs."""
    # common y-limits for the reactivation panels across adn/dg/ca1
    reac_vals = [ufo_reacs_ev[g][s].values for g in groups if s in ufo_reacs_ev[g]]
    if reac_vals:
        reac_ylim = (np.nanmin(np.concatenate([v.ravel() for v in reac_vals])),
                     np.nanmax(np.concatenate([v.ravel() for v in reac_vals])))
    else:
        reac_ylim = None

    for col, g in enumerate(groups):
        ax_reac = fig.add_subplot(gs_reac[0, col])

        if s not in ufo_reacs_ev[g]:
            ax_reac.set_visible(False)
            continue

        # — reactivation traces, one line per eigenvector —
        df = ufo_reacs_ev[g][s]
        for ev_idx in df.columns:
            ax_reac.plot(df.index, df[ev_idx], label=f'EV{ev_idx + 1}', alpha=0.8)
        ax_reac.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax_reac.set_title(f'{g.upper()}  {s.split("/")[-1]}', fontsize=8)
        ax_reac.legend(fontsize=6)
        ax_reac.set_xticklabels([])
        if reac_ylim is not None:
            ax_reac.set_ylim(reac_ylim)
        if col == 0:
            ax_reac.set_ylabel("Reactivation", fontsize=7)

        # — stem | per-neuron HD | per-neuron AHV in the shared bottom grid —
        tc = tc_store[g][s]
        ahv_tc = ahv_tc_store[g][s]
        n_neurons = tc.shape[1]

        # nest a n_neurons×3 grid inside this group's slice of the shared bottom grid
        gs_grp = GridSpecFromSubplotSpec(
            n_neurons, 3, subplot_spec=gs_bot[0:n_neurons, col], wspace=0.35, hspace=0.05
        )

        # stem spans all neuron rows in col 0
        ax_stem = fig.add_subplot(gs_grp[:, 0])
        strongest_reac_idx = df.max().idxmax()
        evecs = evecs_store[g][s]
        i = strongest_reac_idx
        w = evecs[:, i]
        y = np.arange(n_neurons)
        ax_stem.hlines(y, 0, w, colors=f'C{i}', linewidth=2, label=f'EV{i + 1}')
        ax_stem.plot(w, y, 'o', color=f'C{i}', markersize=3)
        ax_stem.axvline(0, color='k', linewidth=0.5)
        ax_stem.set_ylim(-0.5, n_neurons - 0.5)
        ax_stem.invert_yaxis()
        ax_stem.set_xlabel("Weight", fontsize=7)
        if col == 0:
            ax_stem.set_ylabel("Neuron", fontsize=7)
        ax_stem.tick_params(labelsize=6)
        ax_stem.legend(fontsize=6)

        # one subplot per neuron — HD tuning curve (polar)
        for n, col_n in enumerate(tc.columns):
            ax = fig.add_subplot(gs_grp[n, 1], projection='polar')
            ax.plot(tc.index.values, tc[col_n].values, lw=0.8, color=f'C{n}')
            ax.set_yticks([])
            ax.set_xticks([])
            if n == 0:
                ax.set_title("HD", fontsize=7)

        # one subplot per neuron — AHV tuning curve
        for n, col_n in enumerate(ahv_tc.columns):
            ax = fig.add_subplot(gs_grp[n, 2])
            ax.plot(ahv_tc.index.values, ahv_tc[col_n].values, lw=0.8, color=f'C{n}')
            ax.set_yticks([])
            ax.set_xticks([])
            if n == 0:
                ax.set_title("AHV", fontsize=7)


###############################################################################
# PDF output
###############################################################################
n_groups = len(groups)

pdf_path = os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_reactivation_ADN_DG_aversive.pdf")

with PdfPages(pdf_path) as pdf:

    # ── Page 1: pre vs post mean reactivation per group ─────────────────────
    fig1 = figure(figsize=(5 * n_groups, 5))
    gs1 = GridSpec(1, n_groups, figure=fig1, wspace=0.4)

    for col, g in enumerate(groups):
        ax = fig1.add_subplot(gs1[0, col])
        for ep, color in zip(['pre', 'post'], ['C0', 'C1']):
            traces = []
            for s in ufo_reacs[g]:
                if ep not in ufo_reacs[g][s]:
                    continue
                tmp = ufo_reacs[g][s][ep].apply(zscore)
                ax.plot(tmp, color=color, alpha=0.3, linewidth=0.8)
                traces.append(tmp.values.flatten())
            if traces:
                min_len = min(len(z) for z in traces)
                mean_trace = np.nanmean([z[:min_len] for z in traces], 0)
                t = tmp.index.values[:min_len]
                ax.plot(t, mean_trace, color=color, linewidth=2, label=ep)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(g.upper())
        ax.set_xlabel("Time from UFO (s)")
        ax.set_ylim(-5, 5)
        ax.legend(fontsize=7)

    fig1.axes[0].set_ylabel("Reactivation (z-score)")
    tight_layout()
    pdf.savefig(fig1)
    close(fig1)

    # ── Pages 2+: pre vs post per session ───────────────────────────────────
    all_sessions = sorted(set(s for g in groups for s in ufo_reacs[g]))
    for s in all_sessions:
        fig_s = figure(figsize=(5 * n_groups, 4))
        gs_s = GridSpec(1, n_groups, figure=fig_s, wspace=0.4)
        fig_s.suptitle(s.split('/')[-1], fontsize=9)

        for col, g in enumerate(groups):
            ax = fig_s.add_subplot(gs_s[0, col])
            if s not in ufo_reacs[g]:
                ax.set_visible(False)
                continue
            for ep, color in zip(['pre', 'post'], ['C0', 'C1']):
                if ep not in ufo_reacs[g][s]:
                    continue
                tmp = ufo_reacs[g][s][ep].apply(zscore)
                ax.plot(tmp, color=color, linewidth=1.5, label=ep)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            ax.set_title(g.upper(), fontsize=8)
            ax.set_xlabel("Time from UFO (s)", fontsize=7)
            ax.set_ylim(-5, 5)
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel("Reactivation (z-score)", fontsize=7)

        tight_layout()
        pdf.savefig(fig_s)
        close(fig_s)

# show()