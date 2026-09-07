import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpecFromSubplotSpec

from functions.functions import load_mean_waveforms
from ufo_detection import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans
from scipy import stats

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

maxchs = []
ch_pos = {}

ccs = {"sws": [], "wake": []}
metadatas = []
tuning_curves = {"hd":[], "ahv":[]}

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

    location = spikes.location.copy()

    # Separate CA1 and DG neurons based on maxch
    location = spikes.location.copy()
    maxch_values = spikes.maxch[location == "hpc"].values.reshape(-1, 1)
    idx = np.where(location == "hpc")[0]
    if len(maxch_values) > 0:
        clu = KMeans(n_clusters=2, random_state=0).fit(maxch_values)
        map_ = {np.argmin(clu.cluster_centers_.flatten()): "ca1", np.argmax(clu.cluster_centers_.flatten()): "dg"}
        for i, label in enumerate(clu.labels_):
            location[idx[i]] = map_[label]
        # location[(spikes.maxch<30) & (spikes.location == "hpc")] = "ca1"
        # location[(spikes.maxch>=30) & (spikes.location == "hpc")] = "dg"
        spikes.location = location
    groups = spikes.metadata[['location', 'maxch']].groupby("location").groups
    ch_pos[s] = {loc: spikes.maxch[groups[loc]].values for loc in groups.keys()}

    spikes = spikes[(spikes.location == 'adn') | (spikes.group == 0)]

    spikes = spikes[spikes.rate > 1.0]

    ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()

    new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)


    ###############################################################################################
    # TUNING CURVES
    ###############################################################################################
    names = [basename + "_" + str(n) for n in spikes.index]
    cc_ufo_sws = nap.compute_eventcorrelogram(
        spikes, ufo_ts, 0.01, 0.2, sws_ep, norm=True
    )
    cc_ufo_sws.columns = names
    cc_ufo_wake = nap.compute_eventcorrelogram(
        spikes, ufo_ts, 0.01, 0.2, wake_ep, norm=True
    )
    cc_ufo_wake.columns = names

    metadata = spikes.metadata.copy()
    metadata.index = names
    metadata["session"] = basename

    metadatas.append(metadata)
    ccs["sws"].append(cc_ufo_sws)
    ccs["wake"].append(cc_ufo_wake)

    hd_tc = nap.compute_tuning_curves(
        spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'], epochs=new_wake_ep,
        return_pandas=True
    )
    hd_tc.columns = names
    ahv_tc = nap.compute_tuning_curves(
        spikes, ahv, bins=20, range=(-np.pi, np.pi), feature_names=['ahv'], epochs=new_wake_ep, return_pandas=True
    )
    ahv_tc.columns = names

    tuning_curves['hd'].append(hd_tc)
    tuning_curves['ahv'].append(ahv_tc)


###############################################################################################
# AGGREGATE
###############################################################################################
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ccs["sws"] = pd.concat(ccs["sws"], axis=1)
ccs["wake"] = pd.concat(ccs["wake"], axis=1)
metadata = pd.concat(metadatas)
tuning_curves['hd'] = pd.concat(tuning_curves['hd'], axis=1)
tuning_curves['ahv'] = pd.concat(tuning_curves['ahv'], axis=1)

locations = ["adn", "ca1", "dg"]
loc_labels = {"adn": "ADN", "ca1": "CA1", "dg": "DG"}
epoch_keys = ["sws", "wake"]
epoch_labels = {"sws": "SWS", "wake": "Wake"}
colors = {"adn": "#e07b39", "ca1": "#4878d0", "dg": "#6acc65"}

T = ccs["sws"].index.values
sessions = metadata["session"].unique()
n_sess = len(sessions)

# global symmetric colormap scale so all imshows are comparable
global_vmax = max(np.abs(ccs["sws"].values - 1).max(), 0.01)


def draw_row(fig, gs_row, row_label, ep="sws", is_last_row=False):
    """Fill one GridSpec row: 3 locations × (line on top, imshow on bottom)."""
    gs2 = GridSpecFromSubplotSpec(
        2, len(locations), subplot_spec=gs_row,
        wspace=0.35, hspace=0.08, height_ratios=[1, 1]
    )
    for j, loc in enumerate(locations):
        sub = ccs[ep].loc[:, row_label[loc]].copy() if row_label[loc] is not None else None

        ax_l = fig.add_subplot(gs2[0, j])
        ax_i = fig.add_subplot(gs2[1, j], sharex=ax_l)

        if sub is not None and sub.shape[1] > 0:
            mn  = sub.mean(axis=1).values
            std = sub.std(axis=1).values
            ax_l.fill_between(T, mn - std, mn + std, color=colors[loc], alpha=0.25)
            ax_l.plot(T, mn, color=colors[loc], lw=1.5,
                      label=f"n={sub.shape[1]}")

            # sort neurons by peak for imshow
            peak_order = np.argsort(np.argmax(sub.values, axis=0))
            sub = sub.apply(stats.zscore, axis=0)
            mat = sub.values[:, peak_order].T
            ax_i.imshow(
                mat, aspect="auto",
                extent=[T[0], T[-1], mat.shape[0] - 0.5, -0.5],
                origin="upper", cmap="RdBu_r",
                # vmin=1 - global_vmax, vmax=1 + global_vmax,
                # interpolation="nearest",
            )

        ax_l.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax_l.axhline(1, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax_l.set_xlim(T[0], T[-1])
        ax_l.tick_params(labelbottom=False, labelsize=6)
        ax_l.spines[["top", "right"]].set_visible(False)
        ax_l.legend(fontsize=6, frameon=False, loc="upper right")

        ax_i.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax_i.set_xlim(T[0], T[-1])
        ax_i.tick_params(labelsize=6)
        ax_i.spines[["top", "right"]].set_visible(False)

        ax_i.set_xlabel("Time from UFO (s)", fontsize=7)

        # labels only where needed
        if j == 0:
            ax_l.set_ylabel("Norm. rate", fontsize=7)
            ax_i.set_ylabel("Neurons", fontsize=7)

        else:
            plt.setp(ax_i.get_xticklabels(), visible=False)

        ax_l.set_title(loc_labels[loc], fontsize=8, fontweight="bold",
                       color=colors[loc])


###############################################################################################
# PLOTTING
###############################################################################################
fig = plt.figure(figsize=(15, 4 * (n_sess + 1)))
gs = GridSpec(n_sess + 1, 1, hspace=0.55,
              top=0.96, bottom=0.04, left=0.07, right=0.94)

# ── TOP ROW: all sessions ────────────────────────────────────────────────────
all_idx = {
    loc: metadata.index[metadata["location"] == loc].intersection(ccs["sws"].columns)
    for loc in locations
}
draw_row(fig, gs[0, 0], all_idx, is_last_row=(n_sess == 0))
fig.text(0.01, 1 - 0.5 / (n_sess + 1), "All sessions",
         va="center", ha="left", fontsize=8, fontweight="bold", rotation=90)

# ── BOTTOM ROWS: one per session ─────────────────────────────────────────────
for i, sess in enumerate(sessions):
    sess_idx = {
        loc: metadata.index[
            (metadata["session"] == sess) & (metadata["location"] == loc)
        ].intersection(ccs["sws"].columns)
        for loc in locations
    }
    draw_row(fig, gs[i + 1, 0], sess_idx, is_last_row=(i == n_sess - 1))
    fig.text(0.01, 1 - (i + 1.5) / (n_sess + 1), sess,
             va="center", ha="left", fontsize=7, fontweight="bold", rotation=90)

fig.suptitle("Mean cross-correlogram with UFOs — SWS", fontsize=11, fontweight="bold")
plt.savefig(
    os.path.expanduser("~/Dropbox/UFOPhysio/figures/B51_CC_UFO_sws.pdf"),
    bbox_inches="tight", dpi=150,
)

# ── WAKE FIGURE ───────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(15, 4 * (n_sess + 1)))
gs2_main = GridSpec(n_sess + 1, 1, hspace=0.55,
                    top=0.96, bottom=0.04, left=0.07, right=0.94)

all_idx_wake = {
    loc: metadata.index[metadata["location"] == loc].intersection(ccs["wake"].columns)
    for loc in locations
}
draw_row(fig2, gs2_main[0, 0], all_idx_wake, ep="wake", is_last_row=(n_sess == 0))
fig2.text(0.01, 1 - 0.5 / (n_sess + 1), "All sessions",
          va="center", ha="left", fontsize=8, fontweight="bold", rotation=90)

for i, sess in enumerate(sessions):
    sess_idx_wake = {
        loc: metadata.index[
            (metadata["session"] == sess) & (metadata["location"] == loc)
        ].intersection(ccs["wake"].columns)
        for loc in locations
    }
    draw_row(fig2, gs2_main[i + 1, 0], sess_idx_wake, ep="wake",
             is_last_row=(i == n_sess - 1))
    fig2.text(0.01, 1 - (i + 1.5) / (n_sess + 1), sess,
              va="center", ha="left", fontsize=7, fontweight="bold", rotation=90)

fig2.suptitle("Mean cross-correlogram with UFOs — Wake", fontsize=11, fontweight="bold")
plt.savefig(
    os.path.expanduser("~/Dropbox/UFOPhysio/figures/B51_CC_UFO_wake.pdf"),
    bbox_inches="tight", dpi=150,
)


# Splitting by peak positive or peak negatives
from matplotlib.backends.backend_pdf import PdfPages

tmp = ccs["sws"].apply(stats.zscore, axis=0)
idx = tmp.columns[tmp.max() > 3]
peaks = tmp[idx].idxmax()
hpc_active = metadata.index[(metadata.location != "adn") & (metadata.rate > 2)]
pos_pk_idx = np.intersect1d(peaks[peaks > 0].index, hpc_active)
neg_pk_idx = np.intersect1d(peaks[peaks < 0].index, hpc_active)

N_COLS = 3   # units per row on a page
N_ROWS = 4   # unit rows per page
N_PER_PAGE = N_COLS * N_ROWS


def plot_unit_cell(fig, gs_cell, name):
    loc = metadata.loc[name, "location"]
    c   = colors.get(loc, "k")

    gs2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_cell, wspace=0.45, hspace=0.45)

    # ── HD polar ─────────────────────────────────────────────────────────────
    ax_hd = fig.add_subplot(gs2[0, 0], projection="polar")
    if name in tuning_curves["hd"].columns:
        tc = tuning_curves["hd"][name]
        ax_hd.plot(tc.index, tc.values, color=c, lw=1.2)
        ax_hd.fill(tc.index, tc.values, color=c, alpha=0.15)
        ax_hd.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax_hd.set_xticklabels(["E", "N", "W", "S"], fontsize=5)
        ax_hd.tick_params(labelsize=5)
        ax_hd.set_yticklabels([])
    ax_hd.set_title(f"{name}\n{loc_labels.get(loc, loc)}",
                    fontsize=6, fontweight="bold", color=c, pad=2)

    # ── AHV ──────────────────────────────────────────────────────────────────
    ax_ahv = fig.add_subplot(gs2[1, 0])
    if name in tuning_curves["ahv"].columns:
        tc = tuning_curves["ahv"][name]
        ax_ahv.plot(tc.index, tc.values, color=c, lw=1.2)
        ax_ahv.axvline(0, color="gray", lw=0.7, ls=":")
        ax_ahv.set_xlim(tc.index[0], tc.index[-1])
    ax_ahv.set_xlabel("AHV (rad/s)", fontsize=5)
    ax_ahv.set_ylabel("Rate (Hz)", fontsize=5)
    ax_ahv.tick_params(labelsize=5)
    ax_ahv.spines[["top", "right"]].set_visible(False)

    # ── CC SWS ───────────────────────────────────────────────────────────────
    ax_sws = fig.add_subplot(gs2[0, 1])
    if name in ccs["sws"].columns:
        ax_sws.plot(T, ccs["sws"][name].values, color=c, lw=1.2)
        ax_sws.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax_sws.axhline(1, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax_sws.set_xlim(T[0], T[-1])
    ax_sws.set_title("SWS", fontsize=6, color="steelblue")
    ax_sws.set_ylabel("Norm. rate", fontsize=5)
    ax_sws.tick_params(labelsize=5)
    ax_sws.spines[["top", "right"]].set_visible(False)

    # ── CC Wake ──────────────────────────────────────────────────────────────
    ax_wake = fig.add_subplot(gs2[1, 1])
    if name in ccs["wake"].columns:
        ax_wake.plot(T, ccs["wake"][name].values, color=c, lw=1.2)
        ax_wake.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax_wake.axhline(1, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax_wake.set_xlim(T[0], T[-1])
    ax_wake.set_title("Wake", fontsize=6, color="darkorange")
    ax_wake.set_xlabel("Time from UFO (s)", fontsize=5)
    ax_wake.set_ylabel("Norm. rate", fontsize=5)
    ax_wake.tick_params(labelsize=5)
    ax_wake.spines[["top", "right"]].set_visible(False)


def save_units_pdf(unit_idx, pdf_path, suptitle_base):
    n = len(unit_idx)
    if n == 0:
        return
    total_pages = int(np.ceil(n / N_PER_PAGE))
    with PdfPages(pdf_path) as pdf:
        for page, start in enumerate(range(0, n, N_PER_PAGE)):
            batch = unit_idx[start: start + N_PER_PAGE]
            n_rows_page = int(np.ceil(len(batch) / N_COLS))
            fig = plt.figure(figsize=(N_COLS * 5, n_rows_page * 5))
            gs = GridSpec(n_rows_page, N_COLS, figure=fig,
                          hspace=0.55, wspace=0.45,
                          top=0.93, bottom=0.04, left=0.05, right=0.97)
            for k, name in enumerate(batch):
                plot_unit_cell(fig, gs[k // N_COLS, k % N_COLS], name)
            fig.suptitle(
                f"{suptitle_base}  —  page {page + 1}/{total_pages}"
                f"  (units {start + 1}–{start + len(batch)})",
                fontsize=9, fontweight="bold",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


save_units_pdf(
    pos_pk_idx,
    os.path.expanduser("~/Dropbox/UFOPhysio/figures/B51_CC_UFO_pos_pk.pdf"),
    "Post-UFO excitation (SWS z > 3, peak > 0)",
)
save_units_pdf(
    neg_pk_idx,
    os.path.expanduser("~/Dropbox/UFOPhysio/figures/B51_CC_UFO_neg_pk.pdf"),
    "Post-UFO suppression (SWS z > 3, peak < 0)",
)

plt.show()
