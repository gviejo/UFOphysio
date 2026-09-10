# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2026-09-07
"""
Population incoherence during UFOs (TODO.md item 2).

Coherence here follows Viejo, Skromne Carrasco & Peyrache (2025, bioRxiv
10.1101/2025.09.10.675474): for a pair of neurons, "coherence" between two
conditions is the correlation, across all neuron pairs, between their
pairwise firing-rate correlation in each condition -- i.e. a correlation of
correlations. In that paper the two conditions are wake vs. a sleep stage.
Here the second condition is a lag relative to the UFO: we sweep a window
around each UFO event and, at each lag, ask how correlated the population's
pairwise-correlation structure is with its own structure during wake.

For each lag tau in the peri-UFO window:
  1. Pool the population vector at offset tau from every UFO in the session/
     state (one BIN_SIZE_LAG bin per UFO), and build its neuron x neuron
     Pearson correlation matrix C(tau).
  2. Do the same for a matched non-UFO control set (same total spike count,
     drawn from outside a window around any UFO) to get C_ctrl(tau) -- the
     noise floor from finite-sample correlation-matrix estimation alone.
  3. coherence(tau) = corr(upper_triangle(C(tau)), upper_triangle(R)), where
     R is the reference correlation matrix built from the whole wake epoch.

Sweeping tau gives a coherence-vs-time trace: does the wake-like pairwise
structure collapse at the UFO and recover, and on what timescale, relative
to the matched-control noise floor? Computed separately within each
anatomical group (ADN, LMN, PSB) -- never pooled across regions.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pynapple as nap
import _pickle as cPickle
from matplotlib.pyplot import *
from scipy.stats import zscore

sys.path.append("..")
# NOTE: import the submodule explicitly (not `from detection import *`) -- since the
# detection/ move, detection/ has no __init__.py, so it's a namespace package and a
# wildcard/bare-name import from it silently pulls in nothing (see several sibling
# scripts, e.g. main_pyna_UFO_offset_tc.py, which are currently broken this way).
from detection.ufo_detection import loadUFOs

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

datasets = np.unique(np.hstack([
    np.genfromtxt(os.path.join(data_directory, 'datasets_LMN_ADN.list'), delimiter='\n', dtype=str, comments='#'),
    np.genfromtxt(os.path.join(data_directory, 'datasets_LMN_PSB.list'), delimiter='\n', dtype=str, comments='#'),
]))

###############################################################################################
# PARAMETERS
###############################################################################################
BIN_SIZE_WAKE = 0.02            # bin size (s) for the wake reference matrix R (coarse, stable epoch)
BIN_SIZE_LAG = 0.002             # bin size (s) for the peri-UFO lag sweep -- also the lag step below,
                                  # and the bin size the matched control is drawn from (same resolution,
                                  # so a UFO-locked and a control population vector are comparable)
LAG_WINDOW = 0.04              # sweep +/- this many seconds around each UFO
LAG_STEP = BIN_SIZE_LAG          # one bin at a time
N_LAG_NOMINAL = int(round(2 * LAG_WINDOW / LAG_STEP)) + 1  # 21 points if nothing gets truncated
# NOTE: there is deliberately no fixed LAGS array here. nap.compute_perievent's own relative-lag
# axis (its `.t`) is *not* a fixed function of (LAG_WINDOW, BIN_SIZE) alone -- if any UFO in a
# given session/state sits closer than LAG_WINDOW to that state's epoch edge, the whole shared
# axis for that call can come back shrunk (fewer, narrower-range points) rather than just NaN-
# padding that one trial. So the lag axis is read fresh from each call below, not assumed.
EXCLUSION_WINDOW = 0.05          # exclude candidate control bins within +/- 100ms of any UFO (s)
MIN_UNITS = 5                    # minimum simultaneously recorded units for a session to count
MIN_EVENTS = 20                   # minimum matched UFO events (per session/state/lag) to keep
N_CONTROL_REPEATS = 10              # independent matched-control draws to average per lag --
                                     # a single draw is itself a noisy estimate of coherence_ctrl
SEED = 0

STATES = ['wak', 'rem', 'sws']
REGIONS = ['adn', 'lmn', 'psb']  # coherence is computed separately within each of these

rng = np.random.default_rng(SEED)

###############################################################################################
# FUNCTIONS
###############################################################################################
def _as_array(vectors):
    """Accept either a pynapple TsdFrame or a plain (n_samples, n_neurons) ndarray."""
    return vectors.values if hasattr(vectors, "values") else np.asarray(vectors)


def match_controls(vectors, candidate_total, count, rng):
    """
    For each population vector in `vectors`, draw a non-UFO population vector from
    `count` (via the candidate bins in `candidate_total`, a Tsd of per-bin total
    spike counts that already excludes any window around the UFOs) with the same
    total number of spikes, without reusing a control bin twice.

    Both the UFO events and the candidate bins are grouped by their total spike
    count; within each group, as many candidates are drawn (without replacement)
    as there are UFO events sharing that total. If a total has more UFO events
    than available candidates, the extra events are simply left unmatched (there
    is no fallback to a nearby total).
    """
    totals = _as_array(vectors).sum(1).astype(int)
    totals_series = pd.Series(candidate_total.values.astype(int), index=candidate_total.t)
    groups = totals_series.groupby(totals_series.values).groups  # {total: Index of candidate times}

    chosen_t = []
    kept_idx = []
    for k in np.unique(totals):
        pool = groups.get(k)
        if pool is None or len(pool) == 0:
            continue
        idx_events = np.flatnonzero(totals == k)
        n_take = min(len(idx_events), len(pool))
        if n_take < len(idx_events):
            idx_events = rng.choice(idx_events, size=n_take, replace=False)
        chosen_t.append(rng.choice(pool.to_numpy(), size=n_take, replace=False))
        kept_idx.append(idx_events)

    if not chosen_t:
        return None, np.array([], dtype=int)

    chosen_t = np.concatenate(chosen_t)
    kept_idx = np.concatenate(kept_idx)
    # nap.Ts requires (and will otherwise silently re-sort to) increasing timestamps --
    # apply that same reordering to kept_idx, so row i of ctrl_vectors below still lines
    # up with row i of vectors[kept_idx], not just "some row with the same total"
    order = np.argsort(chosen_t)
    chosen_t = chosen_t[order]
    kept_idx = kept_idx[order]
    ctrl_vectors = nap.Ts(t=chosen_t).value_from(count)
    return ctrl_vectors, kept_idx



def correlation_matrix(vectors):
    """Neuron x neuron Pearson correlation matrix from an (n_samples, n_neurons) matrix,
    after z-scoring each population vector (row) across neurons -- see zscore_rows()."""
    X = _as_array(vectors).astype(float)
    X = zscore(X)
    with np.errstate(invalid='ignore'):
        C = np.corrcoef(X, rowvar=False)
    return C


def upper_tri(C):
    return C[np.triu_indices(C.shape[0], k=1)]


def correlation_of_correlations(C1, C2):
    """
    Pearson correlation between the upper-triangular (pairwise-correlation) entries
    of two same-shaped correlation matrices -- the "coherence" measure of Viejo,
    Skromne Carrasco & Peyrache (2025).
    """
    v1, v2 = upper_tri(C1), upper_tri(C2)
    mask = np.isfinite(v1) & np.isfinite(v2)
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(v1[mask], v2[mask])[0, 1]


###############################################################################################
# MAIN LOOP
###############################################################################################
summary = []  # long format: session, region, state, lag, coherence_ufo, coherence_ctrl, n_events
examples = {}  # region -> (session, lags, coh_ufo, coh_ctrl, n_units) for its largest sws population

for s in datasets:

    print(s)

    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if not os.path.exists(filepath):
        continue

    nwb = nap.load_file(filepath)
    spikes_all = nwb['units']
    spikes_all = spikes_all[spikes_all.rate > 1.0]
    spikes_all = spikes_all[spikes_all.location != "-"]

    if len(spikes_all) < MIN_UNITS:
        continue

    epochs = nwb['epochs']
    wake_ep = epochs[epochs.tags == "wake"]
    sws_ep = nwb['sws']
    rem_ep = nwb['rem']

    ufo_ep, ufo_ts = loadUFOs(path)
    if ufo_ts is None:
        continue

    session_locations = np.unique(spikes_all.location)

    for region in REGIONS:

        if region not in session_locations:
            continue

        # coherence is computed within a single anatomical group at a time, never pooled
        # across regions, so a UFO->ADN correlation can't be diluted/inflated by LMN units
        spikes = spikes_all[spikes_all.location == region]

        if len(spikes) < MIN_UNITS:
            continue

        # reference correlation matrix: pairwise correlation of BIN_SIZE_WAKE-binned firing
        # across the whole wake epoch -- the "canonical" structure tested against below
        wake_count = spikes.count(BIN_SIZE_WAKE, ep=wake_ep)
        R = correlation_matrix(wake_count)

        for state, ep in zip(STATES, [wake_ep, rem_ep, sws_ep]):

            ufo_state = ufo_ts.restrict(ep)
            if len(ufo_state) < MIN_EVENTS:
                continue


            # fine-binned population vectors, just for the peri-UFO lag sweep
            count_lag = spikes.count(BIN_SIZE_LAG, ep)

            perievent = nap.compute_perievent(count_lag, ufo_state, LAG_WINDOW + BIN_SIZE_LAG)
            lags = np.round(perievent.t, 6)  # read fresh -- see NOTE on N_LAG_NOMINAL above
            print(len(lags), "lag points for", region, state, "in session", basename)
            if len(lags) < N_LAG_NOMINAL:
                print(f"    {region} {state}: lag axis shrunk to {len(lags)}/{N_LAG_NOMINAL} points "
                      f"(+/-{abs(lags[0]) * 1000:.0f}ms instead of +/-{LAG_WINDOW * 1000:.0f}ms) "
                      "-- a UFO sits close to this state's epoch edge")
            tensor = np.moveaxis(perievent.values, (1, 0, 2), (0, 2, 1))  # (trial, neuron, lag-bin)
            tensor = tensor[~np.isnan(tensor).any((1, 2))]

            # matched, time-lagged control: for each lag, a non-UFO population vector drawn
            # from outside +/-EXCLUSION_WINDOW of any UFO, matched to that lag's UFO-locked
            # vectors on total spike count, at the same BIN_SIZE_LAG resolution -- so it's a
            # per-lag coherence(tau) trace, not a single flat baseline.
            exclude_ep = nap.IntervalSet(
                start=ufo_state.t - EXCLUSION_WINDOW,
                end=ufo_state.t + EXCLUSION_WINDOW,
            )
            candidate_ep = count_lag.time_support.set_diff(exclude_ep)
            total = nap.Tsd(t=count_lag.t, d=count_lag.values.sum(1), time_support=count_lag.time_support)
            candidate_total = total.restrict(candidate_ep)

            coh_ufo = np.full(len(lags), np.nan)
            coh_ctrl = np.full(len(lags), np.nan)
            coh_ctrl_std = np.full(len(lags), np.nan)
            n_events_per_lag = np.zeros(len(lags), dtype=int)

            for li, lag in enumerate(lags):

                lag_vectors = tensor[:, :, li]  # (n_trials, n_neurons) at this lag -- already
                                                 # this shape after the moveaxis above, no .T needed

                # match_controls draws its controls at random, so a single call is a noisy
                # estimate of the control coherence -- call it N_CONTROL_REPEATS times and
                # average. coh_ufo/lag_vectors_matched just need one (any) successful draw's
                # kept_idx, since which UFO events end up matched barely varies between draws
                # (candidate pools are huge relative to event counts).
                coh_ctrl_repeats = []
                lag_vectors_matched = None
                for _ in range(N_CONTROL_REPEATS):
                    ctrl_vectors, kept_idx = match_controls(lag_vectors, candidate_total, count_lag, rng)
                    if ctrl_vectors is None or len(kept_idx) < MIN_EVENTS:
                        continue
                    if lag_vectors_matched is None:
                        lag_vectors_matched = lag_vectors[kept_idx]
                        n_events_this_lag = len(kept_idx)
                    C_ctrl = correlation_matrix(ctrl_vectors)
                    coh_ctrl_repeats.append(correlation_of_correlations(C_ctrl, R))

                if lag_vectors_matched is None or len(coh_ctrl_repeats) == 0:
                    continue

                C_ufo = correlation_matrix(lag_vectors_matched)
                coh_ufo[li] = correlation_of_correlations(C_ufo, R)
                coh_ctrl[li] = np.mean(coh_ctrl_repeats)
                coh_ctrl_std[li] = np.std(coh_ctrl_repeats)
                n_events_per_lag[li] = n_events_this_lag

            for li, lag in enumerate(lags):
                summary.append({
                    "session": basename,
                    "region": region,
                    "state": state,
                    "lag": lag,
                    "coherence_ufo": coh_ufo[li],
                    "coherence_ctrl": coh_ctrl[li],
                    "coherence_ctrl_std": coh_ctrl_std[li],
                    "n_events": n_events_per_lag[li],
                    "n_units": len(spikes),
                    "n_ufo_total": len(ufo_state),
                })

            if state == "sws" and (region not in examples or len(spikes) > examples[region][4]):
                examples[region] = (basename, lags.copy(), coh_ufo.copy(), coh_ctrl.copy(), len(spikes))

summary = pd.DataFrame(summary)

###############################################################################################
# SAVING
###############################################################################################
datatosave = {
    "summary": summary,
    "examples": examples,
    "bin_size_wake": BIN_SIZE_WAKE,
    "bin_size_lag": BIN_SIZE_LAG,
}
cPickle.dump(
    datatosave,
    open(os.path.expanduser("/home/gviejo/Dropbox/UFOPhysio/data/UFO_population_coherence.pickle"), 'wb')
)

###############################################################################################
# PLOTTING
###############################################################################################
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = os.path.expanduser("/home/gviejo/Dropbox/UFOPhysio/figures/UFO_population_coherence.pdf")
pdf = PdfPages(pdf_path)

REGION_COLORS = {'adn': '#1f77b4', 'lmn': '#ff7f0e', 'psb': '#2ca02c'}

if len(summary):
    for region in REGIONS:
        sub_region = summary[summary.region == region]
        if not len(sub_region):
            continue
        figure(figsize=(12, 4))
        for j, state in enumerate(STATES):
            sub = sub_region[sub_region.state == state]
            subplot(1, len(STATES), j + 1)
            if len(sub):
                g_ufo = sub.groupby("lag")["coherence_ufo"].agg(['mean', 'sem'])
                g_ctrl = sub.groupby("lag")["coherence_ctrl"].agg(['mean', 'sem'])
                x = g_ufo.index.values * 1000
                plot(x, g_ufo['mean'], 'o-', color='#d62728', label="UFO-locked")
                fill_between(x, g_ufo['mean'] - g_ufo['sem'], g_ufo['mean'] + g_ufo['sem'],
                             color='#d62728', alpha=0.2)
                plot(x, g_ctrl['mean'], 'o-', color='#1f77b4', label="matched control")
                fill_between(x, g_ctrl['mean'] - g_ctrl['sem'], g_ctrl['mean'] + g_ctrl['sem'],
                             color='#1f77b4', alpha=0.2)
            axvline(0.0, color='k', linestyle='--', linewidth=1)
            xlabel("Time from UFO (ms)")
            if j == 0:
                ylabel("Coherence with wake")
            title(state)
            if j == len(STATES) - 1:
                legend()
        suptitle(f"{region.upper()} (mean +/- SEM across sessions)")
        tight_layout()
        pdf.savefig()
        close()

    # standardized (paired, across-session) difference, grouped across regions: for each
    # session, UFO minus its own matched control at this lag, then that difference's mean
    # across sessions divided by its SD across sessions -- a Cohen's-d-like effect size per
    # lag. One page, one subplot per state, all regions overlaid so they're directly
    # comparable (unlike raw coherence, this effect size doesn't depend on each region's
    # own population size/firing rate, so pooling regions on one axis is meaningful).
    figure(figsize=(12, 4))
    for j, state in enumerate(STATES):
        sub_state = summary[summary.state == state]
        subplot(1, len(STATES), j + 1)
        for region in REGIONS:
            sub = sub_state[sub_state.region == region]
            if not len(sub):
                continue
            diff = sub.assign(diff=sub["coherence_ufo"] - sub["coherence_ctrl"])
            g_diff = diff.groupby("lag")["diff"].agg(['mean', 'std'])
            standardized = g_diff['mean'] / g_diff['std']
            x = g_diff.index.values * 1000
            plot(x, standardized, 'o-', color=REGION_COLORS[region], label=region.upper())
        axhline(0.0, color='grey', linestyle=':', linewidth=1)
        axvline(0.0, color='k', linestyle='--', linewidth=1)
        xlabel("Time from UFO (ms)")
        if j == 0:
            ylabel("Standardized diff\n(UFO - control) / SD")
        title(state)
        if j == len(STATES) - 1:
            legend()
    suptitle("Standardized UFO vs. control difference, by region")
    tight_layout()
    pdf.savefig()
    close()

pdf.close()
print("Saved", pdf_path)
print(summary)
