import os
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from sklearn.decomposition import PCA
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec

from python.functions.functions import load_mean_waveforms
from python.functions.cell_classification import (
    get_waveform_fs,
    extract_all_waveform_features,
    extract_all_waveforms,
    extract_firing_features,
    classify_interneuron,
    classify_dg_subtype,
)

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

datasets = np.genfromtxt(os.path.join(data_directory, 'datasets_DG.list'), delimiter='\n', dtype=str, comments='#')

channels_separation = np.genfromtxt(os.path.join(data_directory, 'channels_CA1_DG.txt'), delimiter=' ', dtype=str, comments='#')
channels_separation = {a[0].split("/")[-1]: a[1].astype('int') for a in channels_separation}

TROUGH_TO_PEAK_THRESH_MS = 0.6
BURST_INDEX_THRESH = 0.02
DG_MIN_GMM_SEPARATION = 0.1
DG_SUBTYPE_MODEL = "kmeans"

all_features = {}
waveforms_by_cell_type = {"interneuron": [], "non-interneuron": []}
dg_non_int_waveforms = []
dg_non_int_wake_sleep_ratio = []
dg_non_int_keys = []

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

        epochs = nwb['epochs']
        wake_ep = epochs[epochs.tags == "wake"]
        sleep_ep = epochs[epochs.tags == "sleep"]

        waveform_path = os.path.join(path, "kilosort4")
        meanwavef, maxch = load_mean_waveforms(waveform_path)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])

        nwb.close()

    else:
        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        wake_ep = data.epochs['wake']
        sleep_ep = data.epochs['sleep']

        waveform_path = path
        meanwavef, maxch = load_mean_waveforms(waveform_path)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])

    spikes.location = [v.lower() for v in spikes.location.values]

    # Separate CA1 and DG neurons based on maxch (same convention as main_pyna_DG_tunings.py)
    location = spikes.location.copy()
    for i in location.index:
        if spikes.location[i] in ["hpc", "ca1"]:
            if spikes.maxch[i] < channels_separation[basename]:
                location[i] = "ca1"
            else:
                location[i] = "dg"
    spikes.location = location

    spikes = spikes[spikes.group == 0]
    # spikes = spikes[spikes.rate > 1.0]

    ###############################################################################################
    # FEATURE EXTRACTION + CLASSIFICATION
    ###############################################################################################
    fs = get_waveform_fs(waveform_path)
    waveform_features = extract_all_waveform_features(spikes, meanwavef, maxch, fs)
    firing_features = extract_firing_features(spikes, sleep_ep, wake_ep, sleep_ep)

    spikes.cell_type = classify_interneuron(
        waveform_features, firing_features,
        trough_to_peak_thresh_ms=TROUGH_TO_PEAK_THRESH_MS, burst_index_thresh=BURST_INDEX_THRESH
    )

    unit_waveforms = extract_all_waveforms(spikes, meanwavef, maxch)
    for uid, wf in unit_waveforms.items():
        ct = spikes.cell_type[uid]
        if ct in waveforms_by_cell_type:
            normalized_wf = wf / (np.max(np.abs(wf)) + 1e-12)
            waveforms_by_cell_type[ct].append(normalized_wf)
            if ct == "non-interneuron" and spikes.location[uid] == "dg":
                dg_non_int_waveforms.append(normalized_wf)
                dg_non_int_wake_sleep_ratio.append(firing_features.loc[uid, "wake_sleep_ratio"])
                dg_non_int_keys.append((s, uid))

    ###############################################################################################
    # COLLECT FOR POOLED SUMMARY
    ###############################################################################################
    session_df = pd.concat([waveform_features, firing_features], axis=1)
    session_df["location"] = spikes.location
    session_df["cell_type"] = spikes.cell_type
    session_df["maxch"] = spikes.maxch
    session_df["session"] = s
    all_features[s] = session_df

pooled = pd.concat(all_features.values(), axis=0)

#################################################################################################
# PCA on the 2nd derivative of DG non-interneuron waveforms
#################################################################################################
wf_lengths = pd.Series([len(wf) for wf in dg_non_int_waveforms])
common_wf_len = wf_lengths.value_counts().idxmax()
keep_wf_mask = [len(wf) == common_wf_len for wf in dg_non_int_waveforms]

stacked_dg_wf = np.array([wf for wf, keep in zip(dg_non_int_waveforms, keep_wf_mask) if keep])
kept_wake_sleep_ratio = [r for r, keep in zip(dg_non_int_wake_sleep_ratio, keep_wf_mask) if keep]
kept_dg_keys = [key for key, keep in zip(dg_non_int_keys, keep_wf_mask) if keep]
dg_second_deriv = np.diff(stacked_dg_wf, n=2, axis=1)

waveform_pca = PCA(n_components=1)
dg_pc1 = pd.DataFrame({
    "pc1": waveform_pca.fit_transform(dg_second_deriv)[:, 0],
    "wake_sleep_ratio": kept_wake_sleep_ratio,
})
dg_pc1["log_wake_sleep_ratio"] = np.log(dg_pc1["wake_sleep_ratio"].replace(0, np.nan))

#################################################################################################
# DG subtype classification (PC1 + log wake/sleep ratio)
#################################################################################################
dg_pc1["dg_subtype"], dg_subtype_diagnostics = classify_dg_subtype(
    dg_pc1, feature_cols=["pc1", "log_wake_sleep_ratio"], mossy_feature="log_wake_sleep_ratio",
    model=DG_SUBTYPE_MODEL, min_separation=DG_MIN_GMM_SEPARATION
)
dg_subtype_by_key = dict(zip(kept_dg_keys, dg_pc1["dg_subtype"]))


def simplify_cell_type(cell_type, dg_subtype):
    if cell_type == "interneuron":
        return "interneuron"
    if dg_subtype == "mossy cell":
        return "mossy"
    if dg_subtype == "granule cell":
        return "granule"
    return "other"


#################################################################################################
# Write per-session cell classification to each session's folder
#################################################################################################
for s in datasets:
    if s not in all_features:
        continue
    session_df = all_features[s]
    out_df = pd.DataFrame({
        "neuron": session_df.index,
        "cell_type": [
            simplify_cell_type(ct, dg_subtype_by_key.get((s, uid), "na"))
            for uid, ct in zip(session_df.index, session_df["cell_type"])
        ],
    })
    out_path = os.path.join(data_directory, s, "cell_classification.csv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print("Wrote {}".format(out_path))

#################################################################################################
# QC figure
#################################################################################################
colors_cell_type = {"interneuron": "red", "non-interneuron": "steelblue", "unknown": "gray"}
colors_dg_subtype = {"granule cell": "#2ca02c", "mossy cell": "#ff7f0e", "unclassified": "gray"}


def plot_joint_scatter_with_marginals(fig, subplot_spec, groups, x_col, y_col, xlabel, ylabel, title,
                                       xline=None, yline=None, yscale=None, xlim=None, bins=180):
    """
    groups : dict {label: (df_subset, color)}
    Draws a scatter of x_col vs y_col in a main axis, with marginal histograms of x_col (top)
    and y_col (right), all sharing axes, inside `subplot_spec`.
    """
    gs_inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=subplot_spec, width_ratios=[4, 1],
                                        height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs_inner[1, 0])
    ax_top = fig.add_subplot(gs_inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs_inner[1, 1], sharey=ax_main)

    y_bins = bins
    if yscale == "log":
        all_y = pd.concat([df[y_col] for df, _ in groups.values()])
        all_y = all_y[all_y > 0].dropna()
        y_bins = np.logspace(np.log10(all_y.min()), np.log10(all_y.max()), bins)

    for label, (df, color) in groups.items():
        ax_main.scatter(df[x_col], df[y_col], s=8, alpha=0.6, color=color,
                         label="{} (n={})".format(label, len(df)))
        ax_top.hist(df[x_col].dropna(), bins=bins, color=color, alpha=0.5, histtype='stepfilled')
        y_vals = df[y_col].dropna()
        if yscale == "log":
            y_vals = y_vals[y_vals > 0]
        ax_right.hist(y_vals, bins=y_bins, color=color, alpha=0.5, histtype='stepfilled',
                       orientation='horizontal')

    if xline is not None:
        ax_main.axvline(xline, color='k', linestyle='--')
        ax_top.axvline(xline, color='k', linestyle='--')
    if yline is not None:
        ax_main.axhline(yline, color='k', linestyle='--')
        ax_right.axhline(yline, color='k', linestyle='--')
    if yscale:
        ax_main.set_yscale(yscale)
        ax_right.set_yscale(yscale)
    if xlim:
        ax_main.set_xlim(*xlim)

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.legend(fontsize=7, frameon=False)
    ax_top.set_title(title)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    return ax_main, ax_top, ax_right


def plot_mean_waveforms(ax, groups, title, ylabel="Norm. amplitude"):
    """
    groups : dict {label: (waveforms, color)}, where `waveforms` is a list of 1D
        np.ndarray (peak-normalized single-channel waveforms, possibly of differing
        length across sessions). Plots mean +/- SEM of the most common length per group,
        overlaid on a single axis.
    """
    for label, (waveforms, color) in groups.items():
        if len(waveforms) == 0:
            continue
        lengths = pd.Series([len(wf) for wf in waveforms])
        common_len = lengths.value_counts().idxmax()
        stacked = np.array([wf for wf in waveforms if len(wf) == common_len])

        m = stacked.mean(axis=0)
        sem = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
        x = np.arange(common_len)
        ax.plot(x, m, color=color, label="{} (n={})".format(label, stacked.shape[0]))
        ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)


def plot_scatter(ax, groups, x_col, y_col, xlabel, ylabel, title):
    """
    groups : dict {label: (df_subset, color)}
    Plain scatter of x_col vs y_col (no marginals), colored per group.
    """
    for label, (df, color) in groups.items():
        ax.scatter(df[x_col], df[y_col], s=8, alpha=0.6, color=color,
                   label="{} (n={})".format(label, len(df)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)


def plot_hist(ax, groups, col, xlabel, title, bins=30, log_x=False):
    """
    groups : dict {label: (df_subset, color)}
    Overlaid step histograms of `col`, one per group. If log_x, bins are log-spaced
    and the x-axis is log-scaled (values <= 0 are dropped, since they have no log).
    """
    if log_x:
        all_values = pd.concat([df[col] for df, _ in groups.values()])
        all_values = all_values[all_values > 0].dropna()
        bins = np.logspace(np.log10(all_values.min()), np.log10(all_values.max()), bins)

    for label, (df, color) in groups.items():
        values = df[col].dropna()
        if log_x:
            values = values[values > 0]
        ax.hist(values, bins=bins, density=True, histtype='step', color=color,
                label="{} (n={})".format(label, len(values)))
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)


pdf_path = "/home/gviejo/Dropbox/UFOPhysio/figures/Cell_classification_DG.pdf"

with PdfPages(pdf_path) as pdf:
    #############################################################################################
    # Pooled page: interneuron classification + DG non-interneuron waveform/firing summaries
    #############################################################################################
    fig = figure(figsize=(18, 13))
    gs = GridSpec(3, 1, hspace=0.4, figure=fig, height_ratios=[1, 1, 1])
    gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.3)
    gs_bottom = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.3)

    groups_it = {
        ct: (pooled[pooled["cell_type"] == ct], color) for ct, color in colors_cell_type.items()
    }
    plot_joint_scatter_with_marginals(
        fig, gs_top[0, 0], groups_it, "trough_to_peak_ms", "burst_index",
        xlabel="Trough-to-peak (ms)", ylabel="Burst index (ACG 3-5ms / 200-300ms)",
        title="Interneuron classification (all sessions pooled)",
        xline=TROUGH_TO_PEAK_THRESH_MS, yline=BURST_INDEX_THRESH, xlim=(0, 2), yscale="log"
    )

    ax_wf = fig.add_subplot(gs_top[0, 1])
    groups_wf = {
        ct: (waveforms_by_cell_type[ct], colors_cell_type[ct])
        for ct in ["interneuron", "non-interneuron"]
    }
    plot_mean_waveforms(ax_wf, groups_wf, "Mean waveforms")

    dg_pool = pooled[(pooled["location"] == "dg") & (pooled["cell_type"] == "non-interneuron")]
    groups_dg_all = {"DG non-interneuron": (dg_pool, colors_cell_type["non-interneuron"])}

    ax_hist_burst = fig.add_subplot(gs_bottom[0, 0])
    plot_hist(ax_hist_burst, groups_dg_all, "burst_index",
              xlabel="Burst index (ACG 3-5ms / 200-300ms)",
              title="Burst index distribution", log_x=True)

    ax_hist_ratio = fig.add_subplot(gs_bottom[0, 1])
    plot_hist(ax_hist_ratio, groups_dg_all, "wake_sleep_ratio",
              xlabel="Wake / sleep rate ratio",
              title="Wake/sleep rate ratio distribution", log_x=True)

    groups_pc1 = {
        subtype: (dg_pc1[dg_pc1["dg_subtype"] == subtype], colors_dg_subtype[subtype])
        for subtype in ["granule cell", "mossy cell", "unclassified"]
    }

    ax_hist_pc1 = fig.add_subplot(gs_bottom[0, 2])
    plot_hist(ax_hist_pc1, groups_pc1, "pc1",
              xlabel="PC1 (2nd derivative of waveform)",
              title="DG non-interneuron\nwaveform PC1 distribution")

    ax_ratio_pc1 = fig.add_subplot(gs_bottom[0, 3])
    plot_scatter(ax_ratio_pc1, groups_pc1, "wake_sleep_ratio", "pc1",
                 xlabel="Wake / sleep rate ratio", ylabel="PC1 (2nd derivative of waveform)",
                 title="DG granule vs mossy cell classification")
    ax_ratio_pc1.set_xscale("log")
    if dg_subtype_diagnostics.get("separation") is not None:
        ax_ratio_pc1.text(0.02, 0.98, "separation={:.2f}".format(dg_subtype_diagnostics["separation"]),
                           transform=ax_ratio_pc1.transAxes, va="top", fontsize=8)

    ax_deriv = fig.add_subplot(gs[2])
    dg_subtype_values = dg_pc1["dg_subtype"].values
    groups_deriv = {
        subtype: (list(dg_second_deriv[dg_subtype_values == subtype]), colors_dg_subtype[subtype])
        for subtype in ["granule cell", "mossy cell"]
    }
    plot_mean_waveforms(ax_deriv, groups_deriv, "Mean 2nd derivative of waveform (DG granule vs mossy cells)",
                         ylabel="2nd derivative (a.u.)")

    pdf.savefig(fig, bbox_inches='tight')
    close(fig)

print("Saved {}".format(pdf_path))