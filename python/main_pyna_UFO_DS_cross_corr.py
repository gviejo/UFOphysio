# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-18 17:59:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-27 12:45:33
# %%
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import yaml
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from functions.functions import load_mean_waveforms
from detection import *
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

with open(os.path.join(data_directory, 'epochs_addg.yaml'), 'r') as f:
    epochs_addg = yaml.safe_load(f)

cc_short = {}
cc_long = {}
perievent = {}

cc_short2 = {}
cc_long2 = {}
perievent2 = {}

all_cc_ufo_dg = {}

ufo_sound_ccs = {}

ratios = []
ratios_click = []

RATIO_WINDOW = 0.1  # window (s) used for the ufo/ds co-occurrence ratios below


def proportion_within(ref_t, query_t, window):
    lo, hi = window
    idx_start = np.searchsorted(query_t, ref_t + lo, side='left')
    idx_end = np.searchsorted(query_t, ref_t + hi, side='right')
    return np.mean((idx_end - idx_start) > 0)


for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sleep_ep = data.epochs['sleep']
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

    # Separating cross corrs in click vs non click.
    events = np.array(epochs_addg[s])
    all_ep = nap.IntervalSet(start=np.sort(np.hstack([sleep_ep.start, wake_ep.start])),
                             end=np.sort(np.hstack([sleep_ep.end, wake_ep.end])), metadata={"tags": events})

    click_ep, click_event = None, None
    if os.path.exists(os.path.join(path, "ttl_sound.npz")):
        ttl = nap.load_file(os.path.join(path, "ttl_sound.npz"))
        click_ep = all_ep[all_ep.tags == "Click"]
        click_event = ttl.restrict(sws_ep).restrict(click_ep)
        assert len(ttl) > 0, f"No TTL events found in {s} for click epochs"

    # sws_ep = sws_ep.intersect(all_ep[all_ep.tags == "Sleep"])
    sws_ep = sws_ep.intersect(sleep_ep[0])


    if ufo_ts is not None and ds_ts is not None:
        grp = nap.TsGroup({0:ufo_ts, 1:ds_ts, }, metadata={"evt":np.array(['ufo', 'ds'])})

        ################
        # Normal Sleep
        ################
        ufo_cc = nap.compute_crosscorrelogram(grp, 1, 2000, sws_ep, norm=True)
        cc_long[s] = ufo_cc[(0,1)]
        ufo_cc = nap.compute_crosscorrelogram(grp, 0.01, 0.5, sws_ep, norm=True)
        cc_short[s] = ufo_cc[(0,1)]
        perievent[s] = nap.compute_perievent(ds_ts, ufo_ts.restrict(sws_ep), window=(-100, 100)).to_tsd()

        # Reversed
        ufo_cc = nap.compute_crosscorrelogram(grp, 0.5, 100, sws_ep, norm=True, reverse=True)
        cc_long2[s] = ufo_cc[(1,0)]
        ufo_cc = nap.compute_crosscorrelogram(grp, 0.01, 1, sws_ep, norm=True, reverse=True)
        cc_short2[s] = ufo_cc[(1,0)]
        perievent2[s] = nap.compute_perievent(ufo_ts, ds_ts.restrict(sws_ep), window=(-100, 100)).to_tsd()

        # Spikes cross-corr
        all_cc_ufo_dg[s] = {}
        for state, ep in zip(['wake', 'sws'], [wake_ep, sws_ep]):
            cc = nap.compute_eventcorrelogram(spikes[spikes.group == 0], ufo_ts, 0.001, 0.1, ep, norm=True)
            cc = (cc - cc.mean(0)) / cc.std(0)
            all_cc_ufo_dg[s][state] = cc

        # %%
        # 4 conditions, all within a RATIO_WINDOW-second window:
        # - ufo_ds_after:  proportion of ufos with a ds in  [ufo_t, ufo_t+w]   (ds follows ufo)
        # - ufo_ds_before: proportion of ufos with a ds in  [ufo_t-w, ufo_t]   (ds precedes ufo)
        # - ds_ufo_after:  proportion of ds with a ufo in   [ds_t, ds_t+w]     (ufo follows ds)
        # - ds_ufo_before: proportion of ds with a ufo in   [ds_t-w, ds_t]     (ufo precedes ds)
        try:
            ufo_t = ufo_ts.restrict(sws_ep).t
            ds_t = ds_ts.restrict(sws_ep).t

            ratios.append(pd.DataFrame(
                index = [s.split("/")[-1]],
                data = {
                    "ufo_ds_after": proportion_within(ufo_t, ds_t, (0, RATIO_WINDOW)),
                    "ufo_ds_before": proportion_within(ufo_t, ds_t, (-RATIO_WINDOW, 0)),
                    "ds_ufo_after": proportion_within(ds_t, ufo_t, (0, RATIO_WINDOW)),
                    "ds_ufo_before": proportion_within(ds_t, ufo_t, (-RATIO_WINDOW, 0)),
                }
            ))
        except Exception as e:
            print(f"  skipped ratios for {s}: {e}")

        ##############
        # Click event
        ##############
        if click_event is not None and len(click_event) > 0:
            ufo_sound_cc = nap.compute_eventcorrelogram(grp, click_event, 0.005, 0.2, click_ep, norm=True)
            ufo_sound_cc.columns = ["ufo", "ds"]
            ufo_sound_ccs[s] = ufo_sound_cc

            # Same 4 conditions as above, but conditioned on the click epoch:
            # ufo/ds timestamps restricted to click_ep instead of the whole sws_ep.
            try:
                tmp = nap.IntervalSet(click_event.t, click_event.t + RATIO_WINDOW)
                ufo_t_click = ufo_ts.restrict(tmp).t
                ds_t_click = ds_ts.restrict(tmp).t

                ratios_click.append(pd.DataFrame(
                    index = [s.split("/")[-1]],
                    data = {
                        "ufo_ds_after": proportion_within(ufo_t_click, ds_t_click, (0, RATIO_WINDOW)),
                        "ufo_ds_before": proportion_within(ufo_t_click, ds_t_click, (-RATIO_WINDOW, 0)),
                        "ds_ufo_after": proportion_within(ds_t_click, ufo_t_click, (0, RATIO_WINDOW)),
                        "ds_ufo_before": proportion_within(ds_t_click, ufo_t_click, (-RATIO_WINDOW, 0)),
                    }
                ))
            except Exception as e:
                print(f"  skipped click ratios for {s}: {e}")

cc_long = pd.DataFrame.from_dict(cc_long)
cc_short = pd.DataFrame.from_dict(cc_short)

ratios = pd.concat(ratios)
ratios_click = pd.concat(ratios_click) if len(ratios_click) else pd.DataFrame(columns=ratios.columns)

# %%

ccs = {"long":cc_long, "short":cc_short, "perievent":perievent, "ratios":ratios, "ratios_click":ratios_click}
import _pickle as cPickle
cPickle.dump(ccs, open(os.path.expanduser("/home/gviejo/Dropbox/UFOPhysio/figures/poster/CC_UFO_DS.pickle"), 'wb'))


# %%

cc_long = cc_long.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
cc_short = cc_short.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)


pdf_path = os.path.expanduser("/home/gviejo/Dropbox/UFOPhysio/figures/Cross-corr_UFO_DS.pdf")
pdf = PdfPages(pdf_path)

DS_COLOR = "crimson"
UFO_COLOR = "steelblue"
INDIV_COLOR = "lightgrey"

n_sessions = len(perievent)
ncols = 4
nrows = max(1, int(np.ceil(n_sessions / ncols)))

fig = figure(figsize=(15, 4 + 3 * nrows))
fig.suptitle("DS activity around UFO events (SWS)", fontsize=15, y=0.995, fontweight="bold")
outergs = GridSpec(2, 1, hspace=0.35, height_ratios=[1, 2.5 * nrows])

gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[0], wspace=0.3)
ax = subplot(gs1[0])
plot(cc_long, color=INDIV_COLOR, alpha=0.6, linewidth=1)
plot(cc_long.mean(1), linewidth=3, color=DS_COLOR, label="Mean across sessions")
axvline(0, color='k', linestyle='--', linewidth=1)
xlabel("Time from UFO (s)")
ylabel("DS rate (norm.)")
title(f"Long window (±{cc_long.index.max():.2g} s)")
legend(frameon=False, fontsize=8)

ax = subplot(gs1[1])
plot(cc_short, color=INDIV_COLOR, alpha=0.6, linewidth=1)
plot(cc_short.mean(1), linewidth=3, color=DS_COLOR, label="Mean across sessions")
axvline(0, color='k', linestyle='--', linewidth=1)
xlabel("Time from UFO (s)")
ylabel("DS rate (norm.)")
title(f"Short window (±{cc_short.index.max():.2g} s)")
legend(frameon=False, fontsize=8)

gs2 = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outergs[1], wspace=0.4, hspace=0.6)

for i, s in enumerate(perievent.keys()):
    gs3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs2[i // ncols, i % ncols], height_ratios=[0.35, 0.65], hspace=0.1)

    ax_raster = subplot(gs3[0, 0])
    plot(perievent[s], '|', markersize=3, color=UFO_COLOR)
    ax_raster.set_title(s.split("/")[-1], fontsize=8)
    ax_raster.tick_params(labelbottom=False, labelsize=6)
    ax_raster.set_ylabel("UFO #", fontsize=7)

    ax_cc = subplot(gs3[1, 0], sharex=ax_raster)
    plot(cc_short[s], linewidth=1.5, color=DS_COLOR)
    axvline(0, color='k', linestyle='--', linewidth=0.8)
    xlim(cc_short.index[0], cc_short.index[-1])
    ax_cc.set_xlabel("Time from UFO (s)", fontsize=7)
    ax_cc.set_ylabel("DS rate", fontsize=7)
    ax_cc.tick_params(labelsize=6)

tight_layout()
pdf.savefig(fig, bbox_inches='tight')
close(fig)
# show()

# %%
cc_long2 = pd.DataFrame.from_dict(cc_long2)
cc_short2 = pd.DataFrame.from_dict(cc_short2)

cc_long2 = cc_long2.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
cc_short2 = cc_short2.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)


n_sessions2 = len(perievent2)
nrows2 = max(1, int(np.ceil(n_sessions2 / ncols)))

fig = figure(figsize=(15, 4 + 3 * nrows2))
fig.suptitle("UFO activity around DS events (SWS)", fontsize=15, y=0.995, fontweight="bold")
outergs = GridSpec(2, 1, hspace=0.35, height_ratios=[1, 2.5 * nrows2])

gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[0], wspace=0.3)
ax = subplot(gs1[0])
plot(cc_long2, color=INDIV_COLOR, alpha=0.6, linewidth=1)
plot(cc_long2.mean(1), linewidth=3, color=UFO_COLOR, label="Mean across sessions")
axvline(0, color='k', linestyle='--', linewidth=1)
xlabel("Time from DS (s)")
ylabel("UFO rate (norm.)")
title(f"Long window (±{cc_long2.index.max():.2g} s)")
legend(frameon=False, fontsize=8)

ax = subplot(gs1[1])
plot(cc_short2, color=INDIV_COLOR, alpha=0.6, linewidth=1)
plot(cc_short2.mean(1), linewidth=3, color=UFO_COLOR, label="Mean across sessions")
axvline(0, color='k', linestyle='--', linewidth=1)
xlabel("Time from DS (s)")
ylabel("UFO rate (norm.)")
title(f"Short window (±{cc_short2.index.max():.2g} s)")
legend(frameon=False, fontsize=8)

gs2 = GridSpecFromSubplotSpec(nrows2, ncols, subplot_spec=outergs[1], wspace=0.4, hspace=0.6)

for i, s in enumerate(perievent2.keys()):
    gs3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs2[i // ncols, i % ncols], height_ratios=[0.35, 0.65], hspace=0.1)

    ax_raster = subplot(gs3[0, 0])
    plot(perievent2[s], '|', markersize=3, color=DS_COLOR)
    ax_raster.set_title(s.split("/")[-1], fontsize=8)
    ax_raster.tick_params(labelbottom=False, labelsize=6)
    ax_raster.set_ylabel("DS #", fontsize=7)

    ax_cc = subplot(gs3[1, 0], sharex=ax_raster)
    plot(cc_short2[s], linewidth=1.5, color=UFO_COLOR)
    axvline(0, color='k', linestyle='--', linewidth=0.8)
    xlim(cc_short2.index[0], cc_short2.index[-1])
    ax_cc.set_xlabel("Time from DS (s)", fontsize=7)
    ax_cc.set_ylabel("UFO rate", fontsize=7)
    ax_cc.tick_params(labelsize=6)

tight_layout()
pdf.savefig(fig, bbox_inches='tight')
close(fig)

# %%
# UFO and DS event-correlograms triggered by the click sound, one panel per session
n_sessions3 = len(ufo_sound_ccs)
nrows3 = max(1, int(np.ceil(n_sessions3 / ncols)))

fig = figure(figsize=(15, 3 * nrows3 + 1))
fig.suptitle("UFO / DS rate around click-sound onset", fontsize=15, y=0.995, fontweight="bold")
gs = GridSpec(nrows3, ncols, wspace=0.35, hspace=0.6)

for i, s in enumerate(ufo_sound_ccs.keys()):
    ax = subplot(gs[i // ncols, i % ncols])
    cc = ufo_sound_ccs[s]
    plot(cc["ufo"], linewidth=1.5, color=UFO_COLOR, label="UFO")
    plot(cc["ds"], linewidth=1.5, color=DS_COLOR, label="DS")
    axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_title(s.split("/")[-1], fontsize=8)
    ax.set_xlabel("Time from click (s)", fontsize=7)
    ax.set_ylabel("Rate (norm.)", fontsize=7)
    ax.tick_params(labelsize=6)
    if i == 0:
        ax.legend(frameon=False, fontsize=7)

tight_layout()
pdf.savefig(fig, bbox_inches='tight')
close(fig)




# %%
# Spikes cross-corr

# fig = figure(figsize=(8,30))
#
# gs = GridSpec(len(all_cc_ufo_dg), 2, wspace=0.3, hspace=0.5)
#
# for i, s in enumerate(all_cc_ufo_dg.keys()):
#
#     for j, state in enumerate(['wake', 'sws']):
#
#         gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i, j], wspace=0.3, hspace=0.3)
#
#         ax = subplot(gs2[0, 0])
#         imshow(all_cc_ufo_dg[s][state].T, aspect='auto')
#         title(f"{s.split('/')[-1]} - {state}")
#
#         ax = subplot(gs2[1,0])
#         plot(all_cc_ufo_dg[s][state], alpha=0.5)
#         plot(all_cc_ufo_dg[s][state].mean(1), linewidth=2)
#         xlabel("Time UFO (s)")
#         axvline(0)
#         # title(s.split("/")[-1])
#
# tight_layout()
# pdf.savefig(fig)
# close(fig)

# %%
# Proportion of UFO/DS co-occurrences within RATIO_WINDOW seconds, 4 conditions:
# - ufo_ds_after / ufo_ds_before: relative to ufos (ds follows / precedes)
# - ds_ufo_after / ds_ufo_before: relative to ds (ufo follows / precedes)
cols = ["ufo_ds_after", "ufo_ds_before", "ds_ufo_after", "ds_ufo_before"]
labels = ["DS after\nUFO", "DS before\nUFO", "UFO after\nDS", "UFO before\nDS"]
colors = ["skyblue", "steelblue", "salmon", "indianred"]


def add_sig_bracket(xa, xb, y, pval):
    plot([xa, xa, xb, xb], [y, y * 1.05, y * 1.05, y], color='k')
    sig = "n.s." if pval >= 0.05 else "*" if pval >= 0.01 else "**" if pval >= 0.001 else "***"
    text((xa + xb) / 2, y * 1.06, sig, ha='center', fontsize=8)


def plot_ratio_bars(ax, ratios_df, page_title):
    sca(ax)
    x = np.arange(len(cols))
    means = ratios_df[cols].mean().values
    sems = ratios_df[cols].sem().values

    bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.8, zorder=1)
    xticks(x, labels, fontsize=8)

    for _, row in ratios_df.iterrows():
        plot(x[:2], row[["ufo_ds_after", "ufo_ds_before"]], color="grey", alpha=0.4, marker='o', markersize=3, zorder=2)
        plot(x[2:], row[["ds_ufo_after", "ds_ufo_before"]], color="grey", alpha=0.4, marker='o', markersize=3, zorder=2)

    ylabel("Proportion")
    y_max = ratios_df[cols].values.max()

    pval_ufo = stats.wilcoxon(ratios_df["ufo_ds_after"], ratios_df["ufo_ds_before"]).pvalue
    add_sig_bracket(x[0], x[1], y_max * 1.1, pval_ufo)

    pval_ds = stats.wilcoxon(ratios_df["ds_ufo_after"], ratios_df["ds_ufo_before"]).pvalue
    add_sig_bracket(x[2], x[3], y_max * 1.1, pval_ds)

    ylim(0, y_max * 1.25)
    ax.set_title(f"{page_title} (n={len(ratios_df)})\nUFO: p={pval_ufo:.3g}   DS: p={pval_ds:.3g}", fontsize=9)


fig, axes = subplots(1, 2, figsize=(10, 4.5))
plot_ratio_bars(axes[0], ratios, "Baseline (whole SWS epoch)")
if len(ratios_click):
    plot_ratio_bars(axes[1], ratios_click, "Conditioned on click epochs")
else:
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, "No click data", ha='center', va='center')

fig.suptitle(
    f"Wilcoxon signed-rank test, paired by session ({RATIO_WINDOW}s window):\n"
    "UFO group tests DS-after-UFO vs DS-before-UFO (÷ n UFO); "
    "DS group tests UFO-after-DS vs UFO-before-DS (÷ n DS)",
    fontsize=8, y=1.06
)

tight_layout()
pdf.savefig(fig, bbox_inches='tight')
close(fig)

pdf.close()