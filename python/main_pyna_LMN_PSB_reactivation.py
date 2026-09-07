import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import zscore

from functions.functions import load_mean_waveforms
from ufo_detection import *
from reactivation import compute_reactivation
from matplotlib.pyplot import *

def read_neuroscope_intervals(path, basename, name):
    """
    """
    path2file = os.path.join(path, basename + "." + name + ".evt")
    # df = pd.read_csv(path2file, delimiter=' ', usecols = [0], header = None)
    tmp = np.genfromtxt(path2file)[:, 0]
    df = tmp.reshape(len(tmp) // 2, 2)
    isets = nap.IntervalSet(df[:, 0], df[:, 1], time_units="ms")
    return isets

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')

groups = ['lmn', 'psb']

ufo_reacs = {g: {} for g in groups}
ufo_reacs_ctrl = {g: {} for g in groups}
ufo_reacs_ev = {g: {} for g in groups}  # per-eigenvector perievent, [g][s] = DataFrame (time x n_evecs)
evecs_store = {g: {} for g in groups}   # significant eigenvectors, [g][s] = array (n_neurons x n_evecs)
tc_store = {g: {} for g in groups}      # HD tuning curves, [g][s] = DataFrame (bins x neurons)
ahv_tc_store = {g: {} for g in groups}  # AHV tuning curves, [g][s] = DataFrame (bins x neurons)
weighted_tc_store = {g: {} for g in groups}      # weighted-avg HD TC per evec, [g][s] = list of arrays (n_bins,)
weighted_ahv_tc_store = {g: {} for g in groups}  # weighted-avg AHV TC per evec, [g][s] = list of arrays (n_bins,)
ufo_reacs_up = {g: {} for g in groups}  # reactivation at up time, [g][s] = DataFrame (time x 1)


for s in datasets:
# for s in ['LMN-PSB/B2000/B2001/B2001-210901']:
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

        nwb.close()

    else:

        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')

    try:
        up_ep = read_neuroscope_intervals(path, basename, 'up')
    except:
        up_ep = None


    spikes.location = [v.lower() for v in spikes.location.values]
    ufo_ep, ufo_ts = loadUFOs(path)

    spikes = spikes[spikes.rate > 1.0]
    spikes = spikes[(spikes.location == "lmn") | (spikes.location == "psb")]

    # Filtering out non-HD LMN neurons
    grp_spikes = spikes[spikes.location == 'lmn']
    lmn_idx = grp_spikes.index
    tc2 = nap.compute_tuning_curves(
        grp_spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'],
        epochs=wake_ep
    )
    SI = nap.compute_mutual_information(tc2)['bits/spike']
    grp_spikes = grp_spikes[SI > 0.1]
    lmn_idx_filt = grp_spikes.index
    else_idx = np.setdiff1d(spikes.index, lmn_idx)
    tokeep = np.sort(np.hstack([else_idx, lmn_idx_filt]))
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

    # if (np.sum(spikes.location == "psb") > 5) and (np.sum(spikes.location == "lmn") > 5):
    if np.sum(spikes.location == "lmn") > 5:

        bin_size = 0.01

        for g in groups:
            grp_spikes = spikes[spikes.location == g]

            if len(grp_spikes) < 3:
                continue

            tc_store[g][s] = tuning_curves[grp_spikes.index]
            ahv_tc_store[g][s] = ahv_tc[grp_spikes.index]

            ##################################################
            # Compute reactivation
            ##################################################
            significant_evecs, significant_evals = compute_reactivation(grp_spikes, new_wake_ep, bin_size)
            evecs_store[g][s] = significant_evecs

            # Probability of participating = evec**2 (unit eigenvectors already sum to 1 when squared)
            wtcs, wahv_tcs = [], []
            for i in range(significant_evecs.shape[1]):
                probs = significant_evecs[:, i] ** 2
                probs = probs / np.sum(probs)  # Normalize to sum to 1
                wtcs.append(tc_store[g][s].values @ probs)
                wahv_tcs.append(ahv_tc_store[g][s].values @ probs)
            weighted_tc_store[g][s] = wtcs
            weighted_ahv_tc_store[g][s] = wahv_tcs

            sleep_count = grp_spikes.count(bin_size, sws_ep)
            sleep_count = sleep_count.smooth(3*bin_size)
            sleep_count = sleep_count - sleep_count.mean(0)
            sleep_count = sleep_count / sleep_count.std(0)

            ufo_ts_sws = ufo_ts.restrict(sws_ep)

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
            ufo_reac = nap.compute_perievent(reactivation, ufo_ts_sws, window=(-0.5, 0.5))
            tmp = pd.DataFrame(np.nanmean(ufo_reac, 1).as_series())
            tmp.columns = [s.split("/")[-1]]
            ufo_reacs[g][s] = tmp

            # Per-eigenvector perievent
            ev_perivent = {}
            for i in range(ev_scores.shape[1]):
                ev_tsd = nap.Tsd(t=sleep_count.t, d=ev_scores[:, i] - ev_scores_ctrl[:, i], time_support=sleep_count.time_support)
                ufo_reac_i = nap.compute_perievent(ev_tsd, ufo_ts_sws, window=(-0.5, 0.5))
                ev_perivent[i] = np.nanmean(ufo_reac_i, 1).as_series()
            ufo_reacs_ev[g][s] = pd.DataFrame(ev_perivent)

            # extra control : random time points during sws_ep
            tmp = []
            for i in range(20):
                random_ts = nap.jitter_timestamps(ufo_ts_sws, 10, keep_tsupport=True)
                ctr2_reac = nap.compute_perievent(reactivation, random_ts, window=(-0.5, 0.5))
                tmp.append(pd.DataFrame(np.nanmean(ctr2_reac, 1).as_series(), columns=[s.split("/")[-1]]))
            tmp = pd.concat(tmp, axis=1)
            ufo_reacs_ctrl[g][s] = pd.DataFrame(tmp.mean(axis=1), columns=[s.split("/")[-1]])

            # Reactivation at up time
            if up_ep is not None:
                up_reac = nap.compute_perievent(reactivation, up_ep.starts, window=(-0.5, 0.5))
                ufo_reacs_up[g][s] = pd.DataFrame(np.nanmean(up_reac, 1).as_series(), columns=[s.split("/")[-1]])

###############################################################################
# Save
###############################################################################
import pickle
save_dir = os.path.expanduser("~/Dropbox/UFOPhysio/data")
os.makedirs(save_dir, exist_ok=True)
ufo_reacs_zscore = {g: {s: ufo_reacs[g][s].apply(zscore) for s in ufo_reacs[g]} for g in groups}
with open(os.path.join(save_dir, "ufo_reacs_LMN_PSB.pkl"), "wb") as f:
    pickle.dump(ufo_reacs_zscore, f)

###############################################################################
# Helpers
###############################################################################
def draw_session_page(fig, s, gs_reac, gs_bot, n_max):
    """Draw all groups for one session into pre-built reactivation and bottom gridspecs."""
    # common y-limits for the reactivation panels across lmn/psb
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
all_sessions = sorted(set(s for g in groups for s in ufo_reacs_ev[g].keys()))
n_groups = len(groups)

pdf_path = os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_reactivation_LMN_PSB.pdf")

with PdfPages(pdf_path) as pdf:

    # ── Page 1: mean reactivation summary ───────────────────────────────────
    fig1 = figure(figsize=(5 * n_groups, 5))
    gs1 = GridSpec(1, n_groups, figure=fig1, wspace=0.4)
    summary_axes = [fig1.add_subplot(gs1[0, col]) for col in range(n_groups)]

    for ax, g in zip(summary_axes, groups):
        real_traces = []
        for s in ufo_reacs[g].keys():
            tmp = ufo_reacs[g][s].apply(zscore)
            ax.plot(tmp, alpha=0.5)
            real_traces.append(tmp.values.flatten())
        if real_traces:
            min_len = min(len(z) for z in real_traces)
            real_mean = np.nanmean([z[:min_len] for z in real_traces], 0)
            t = ufo_reacs[g][s].index.values[:min_len]
            ax.plot(t[:min_len], real_mean, color='k', linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(g.upper())
        ax.set_xlabel("Time from UFO (s)")
        ax.set_ylim(-5, 5)

    summary_axes[0].set_ylabel("Reactivation (z-score)")
    tight_layout()
    pdf.savefig(fig1)
    close(fig1)

    # ── Pages 2+: one page per session ──────────────────────────────────────
    for s in all_sessions:
        n_max = max(
            (tc_store[g][s].shape[1] for g in groups if s in tc_store[g]),
            default=1
        )
        fig_h = 4 + n_max * 1.5
        fig_s = figure(figsize=(7 * n_groups, fig_h))

        # outer split: reactivation (top) | stem/TC/AHV (bottom)
        gs_outer = GridSpec(2, 1, figure=fig_s, height_ratios=[2, n_max], hspace=0.35)
        gs_reac = GridSpecFromSubplotSpec(1, n_groups, subplot_spec=gs_outer[0], wspace=0.4)
        gs_bot  = GridSpecFromSubplotSpec(n_max, n_groups, subplot_spec=gs_outer[1], wspace=0.4, hspace=0.05)

        draw_session_page(fig_s, s, gs_reac, gs_bot, n_max)

        tight_layout()
        pdf.savefig(fig_s)
        close(fig_s)

    # ── Pages: Weighted-average tuning curves per eigenvector, per group ────────
    for g in groups:
        sessions_g = sorted(weighted_tc_store[g].keys())
        if not sessions_g:
            continue

        n_sessions = len(sessions_g)
        max_evecs = max(len(weighted_tc_store[g][s]) for s in sessions_g)
        n_cols = max_evecs * 2  # alternating: HD polar (even cols) | AHV linear (odd cols)

        fig_w = figure(figsize=(max(max_evecs * 3, 6), n_sessions * 2.5 + 1))
        fig_w.suptitle(
            f'{g.upper()} – Weighted-average tuning curves per eigenvector\n'
            f'(weights = evec² / Σevec² = probability of participating)',
            fontsize=9
        )
        gs_w = GridSpec(n_sessions, n_cols, figure=fig_w, wspace=0.5, hspace=0.8)

        for row, s in enumerate(sessions_g):
            wtcs = weighted_tc_store[g][s]
            wahv = weighted_ahv_tc_store[g][s]
            tc_ref = tc_store[g][s]
            ahv_ref = ahv_tc_store[g][s]

            for ev_i in range(len(wtcs)):
                # HD polar plot
                ax_hd = fig_w.add_subplot(gs_w[row, ev_i * 2], projection='polar')
                ax_hd.plot(tc_ref.index.values, wtcs[ev_i], lw=1.2, color=f'C{ev_i}')
                ax_hd.set_yticks([])
                ax_hd.set_xticks([])
                if row == 0:
                    ax_hd.set_title(f'EV{ev_i + 1}\nHD', fontsize=7)

                # AHV linear plot
                ax_ahv = fig_w.add_subplot(gs_w[row, ev_i * 2 + 1])
                ax_ahv.plot(ahv_ref.index.values, wahv[ev_i], lw=1.2, color=f'C{ev_i}')
                ax_ahv.axvline(0, color='k', linewidth=0.5)
                ax_ahv.set_yticks([])
                if row == 0:
                    ax_ahv.set_title(f'EV{ev_i + 1}\nAHV', fontsize=7)
                if ev_i == 0:
                    ax_ahv.set_ylabel(s.split('/')[-1], fontsize=6)
                if row < n_sessions - 1:
                    ax_ahv.set_xticklabels([])

        tight_layout()
        pdf.savefig(fig_w)
        close(fig_w)

    # ── Last page: real reactivation vs random-jitter control ───────────────
    fig_ctrl = figure(figsize=(5 * n_groups, 5))
    gs_ctrl = GridSpec(1, n_groups, figure=fig_ctrl, wspace=0.4)
    fig_ctrl.suptitle("Reactivation: real UFOs vs random-jitter control", fontsize=10)

    for col, g in enumerate(groups):
        ax = fig_ctrl.add_subplot(gs_ctrl[0, col])

        real_traces, ctrl_traces = [], []
        for s in ufo_reacs[g].keys():
            real_tmp = ufo_reacs[g][s].apply(zscore)
            ax.plot(real_tmp, color='C0', alpha=0.3, linewidth=0.7)
            real_traces.append(real_tmp.values.flatten())

        for s in ufo_reacs_ctrl[g].keys():
            ctrl_tmp = ufo_reacs_ctrl[g][s].apply(zscore)
            ax.plot(ctrl_tmp, color='C1', alpha=0.3, linewidth=0.7)
            ctrl_traces.append(ctrl_tmp.values.flatten())

        if real_traces:
            min_len = min(len(z) for z in real_traces)
            real_mean = np.nanmean([z[:min_len] for z in real_traces], 0)
            t = ufo_reacs[g][list(ufo_reacs[g].keys())[-1]].index.values[:min_len]
            ax.plot(t, real_mean, color='C0', linewidth=1.5, label='Real UFOs')

        if ctrl_traces:
            min_len_c = min(len(z) for z in ctrl_traces)
            ctrl_mean = np.nanmean([z[:min_len_c] for z in ctrl_traces], 0)
            t_c = ufo_reacs_ctrl[g][list(ufo_reacs_ctrl[g].keys())[-1]].index.values[:min_len_c]
            ax.plot(t_c, ctrl_mean, color='C1', linewidth=1.5, label='Jitter ctrl')

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(g.upper())
        ax.set_xlabel("Time from UFO (s)")
        ax.set_ylim(-5, 5)
        ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("Reactivation (z-score)")

    tight_layout()
    pdf.savefig(fig_ctrl)
    close(fig_ctrl)

    # ── Last page: reactivation at UP-state onset ────────────────────────────
    fig_up = figure(figsize=(5 * n_groups, 5))
    gs_up = GridSpec(1, n_groups, figure=fig_up, wspace=0.4)
    fig_up.suptitle("Reactivation at UP-state onset", fontsize=10)

    for col, g in enumerate(groups):
        ax = fig_up.add_subplot(gs_up[0, col])
        real_traces = []
        for s in ufo_reacs_up[g].keys():
            tmp = ufo_reacs_up[g][s].apply(zscore)
            ax.plot(tmp, alpha=0.5)
            real_traces.append(tmp.values.flatten())
        if real_traces:
            min_len = min(len(z) for z in real_traces)
            real_mean = np.nanmean([z[:min_len] for z in real_traces], 0)
            t = ufo_reacs_up[g][s].index.values[:min_len]
            ax.plot(t, real_mean, color='k', linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(g.upper())
        ax.set_xlabel("Time from UP onset (s)")
        ax.set_ylim(-5, 5)
        if col == 0:
            ax.set_ylabel("Reactivation (z-score)")

    tight_layout()
    pdf.savefig(fig_up)
    close(fig_up)

# show()