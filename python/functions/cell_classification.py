import numpy as np
import pandas as pd
import pynapple as nap
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from python.functions.functions import loadXML


def get_waveform_fs(waveform_path):
    """
    Sampling rate (Hz) of the mean waveforms stored at `waveform_path` — the same
    path passed to `load_mean_waveforms` (either a neurosuite session folder or its
    `kilosort4` subfolder, both of which contain a `.xml` file).
    """
    _, fs, _, _ = loadXML(waveform_path)
    return fs


def compute_waveform_features(waveform, fs, upsample_factor=10):
    """
    Extract shape features from a single-channel mean spike waveform.

    Parameters
    ----------
    waveform : np.ndarray, shape (n_samples,)
        Mean waveform on the max-amplitude channel for one unit.
    fs : float
        Sampling rate (Hz) of `waveform`.
    upsample_factor : int
        Cubic-spline upsampling factor applied before measuring timing, for
        sub-sample precision given the small number of raw samples per waveform.

    Returns
    -------
    dict with trough_to_peak_ms, half_amplitude_width_ms, trough_amp, peak_amp.
    """
    n = len(waveform)
    x = np.arange(n)
    xi = np.linspace(0, n - 1, n * upsample_factor)
    wf = CubicSpline(x, waveform)(xi)
    fs_up = fs * upsample_factor

    trough_idx = np.argmin(wf)
    trough_amp = wf[trough_idx]
    peak_idx = trough_idx + np.argmax(wf[trough_idx:])
    peak_amp = wf[peak_idx]
    trough_to_peak_ms = (peak_idx - trough_idx) / fs_up * 1000.0

    half_amp = trough_amp / 2.0
    below = np.where(wf <= half_amp)[0]
    if len(below) > 1:
        half_amplitude_width_ms = (below[-1] - below[0]) / fs_up * 1000.0
    else:
        half_amplitude_width_ms = np.nan

    return {
        "trough_to_peak_ms": trough_to_peak_ms,
        "half_amplitude_width_ms": half_amplitude_width_ms,
        "trough_amp": trough_amp,
        "peak_amp": peak_amp,
    }


def extract_all_waveform_features(spikes, meanwavef, maxch, fs, upsample_factor=10):
    """
    Compute waveform features for every unit in `spikes`.

    Parameters
    ----------
    spikes : nap.TsGroup
        Used only to map (group, local index) -> global unit id, via
        `spikes.index[spikes.group == g][i]` (same convention as
        `main_pyna_DG_tunings.py`).
    meanwavef : dict {group: np.ndarray (n_units, n_samples, n_channels)}
        From `load_mean_waveforms`.
    maxch : dict {group: np.ndarray (n_units,)}
        Max-amplitude channel per unit, from `load_mean_waveforms`.
    fs : float
        Sampling rate (Hz), from `get_waveform_fs`.

    Returns
    -------
    pd.DataFrame indexed by unit id, columns as in `compute_waveform_features`.
    """
    rows = {}
    for g in meanwavef.keys():
        unit_ids = spikes.index[spikes.group == g]
        for i, uid in enumerate(unit_ids):
            ch = maxch[g][i]
            wf = meanwavef[g][i][:, ch]
            rows[uid] = compute_waveform_features(wf, fs, upsample_factor=upsample_factor)
    return pd.DataFrame(rows).T


def extract_all_waveforms(spikes, meanwavef, maxch):
    """
    Raw (unnormalized) single-channel mean waveform on the max-amplitude channel,
    for every unit in `spikes`. Same (group, local index) -> global unit id mapping
    as `extract_all_waveform_features`.

    Returns dict {unit_id: np.ndarray, shape (n_samples,)}.
    """
    waveforms = {}
    for g in meanwavef.keys():
        unit_ids = spikes.index[spikes.group == g]
        for i, uid in enumerate(unit_ids):
            ch = maxch[g][i]
            waveforms[uid] = meanwavef[g][i][:, ch]
    return waveforms


def compute_burst_index(spikes, ep, binsize_ms=0.5, windowsize_ms=300.0,
                         burst_range_ms=(3.0, 5.0), baseline_range_ms=(200.0, 300.0)):
    """
    Burst index per unit: number of spikes in the `burst_range_ms` bins of the spike
    autocorrelogram divided by the number of spikes in the `baseline_range_ms` bin(s)
    (classic "3-5ms bins / 200-300ms bin" burst-index convention).

    Computed from `nap.compute_autocorrelogram(..., norm=False)`. Those values equal
    raw_bin_spike_count / (n_spikes * binsize) for every bin (same n_spikes/binsize
    factor throughout), so the ratio between the burst and baseline windows is
    identical to the ratio of raw spike counts.

    Returns pd.Series indexed by unit id, name="burst_index".
    """
    binsize_s = binsize_ms / 1000.0
    windowsize_s = windowsize_ms / 1000.0
    acg = nap.compute_autocorrelogram(spikes.restrict(ep), binsize=binsize_s,
                                       windowsize=windowsize_s, ep=ep, norm=False)

    lags = acg.index.values
    burst_mask = (lags >= burst_range_ms[0] / 1000.0) & (lags <= burst_range_ms[1] / 1000.0)
    baseline_mask = (lags >= baseline_range_ms[0] / 1000.0) & (lags <= baseline_range_ms[1] / 1000.0)

    burst = acg.loc[burst_mask].sum(axis=0)
    baseline = acg.loc[baseline_mask].sum(axis=0)
    burst_index = burst / baseline.replace(0, np.nan)
    burst_index.name = "burst_index"
    return burst_index


def compute_isi_cv(spikes, ep, min_n_isis=20):
    """
    Coefficient of variation of each unit's ISIs (within `ep`), computed directly
    from spike times. Units with fewer than `min_n_isis` ISIs -> NaN.

    Returns pd.Series indexed by unit id, name="isi_cv".
    """
    restricted = spikes.restrict(ep)
    values = {}
    for uid in restricted.index:
        isis = np.diff(restricted[uid].t)
        if len(isis) < min_n_isis:
            values[uid] = np.nan
        else:
            values[uid] = isis.std() / (isis.mean() + 1e-12)
    return pd.Series(values, name="isi_cv")


def extract_firing_features(spikes, ep, wake_ep, sleep_ep, binsize_ms=0.5, windowsize_ms=300.0,
                             burst_range_ms=(3.0, 5.0), baseline_range_ms=(200.0, 300.0),
                             min_n_isis=20):
    """
    Returns pd.DataFrame indexed by unit id with columns
    [rate, burst_index, isi_cv, wake_rate, sleep_rate, wake_sleep_ratio].

    `rate`/`burst_index`/`isi_cv` are computed within `ep` (e.g. sleep_ep, for reliable
    burst statistics). `wake_rate`/`sleep_rate`/`wake_sleep_ratio` are computed
    specifically within `wake_ep`/`sleep_ep`, regardless of what `ep` is.
    """
    restricted = spikes.restrict(ep)
    rate = pd.Series(restricted.rates, index=restricted.index, name="rate")
    burst_index = compute_burst_index(
        spikes, ep, binsize_ms=binsize_ms, windowsize_ms=windowsize_ms,
        burst_range_ms=burst_range_ms, baseline_range_ms=baseline_range_ms
    )
    isi_cv = compute_isi_cv(spikes, ep, min_n_isis=min_n_isis)

    wake_restricted = spikes.restrict(wake_ep)
    sleep_restricted = spikes.restrict(sleep_ep)
    wake_rate = pd.Series(wake_restricted.rates, index=wake_restricted.index, name="wake_rate")
    sleep_rate = pd.Series(sleep_restricted.rates, index=sleep_restricted.index, name="sleep_rate")
    wake_sleep_ratio = wake_rate / sleep_rate.replace(0, np.nan)
    wake_sleep_ratio.name = "wake_sleep_ratio"

    return pd.concat([rate, burst_index, isi_cv, wake_rate, sleep_rate, wake_sleep_ratio], axis=1)


def classify_interneuron(waveform_features, firing_features,
                          trough_to_peak_thresh_ms=0.40, burst_index_thresh=1.5):
    """
    Fixed-threshold interneuron classification: narrow waveform
    (trough_to_peak_ms < trough_to_peak_thresh_ms) AND high burst index
    (burst_index > burst_index_thresh) => "interneuron", else "non-interneuron".
    Units missing waveform or burst index data -> "unknown".

    Returns pd.Series (name="cell_type") indexed by the union of both inputs' indices.
    """
    idx = waveform_features.index.union(firing_features.index)
    ttp = waveform_features.reindex(idx)["trough_to_peak_ms"]
    burst_index = firing_features.reindex(idx)["burst_index"]

    label = pd.Series("unknown", index=idx, name="cell_type")
    valid = ttp.notna() & burst_index.notna()
    label[valid] = "non-interneuron"
    is_interneuron = valid & (ttp < trough_to_peak_thresh_ms) & (burst_index < burst_index_thresh)
    label[is_interneuron] = "interneuron"
    return label


def classify_dg_subtype(features, feature_cols=("rate", "burst_index"), mossy_feature=None,
                         model="gmm", min_n_units=10, min_separation=0.3, random_state=0):
    """
    Unsupervised (2 clusters) granule-cell vs mossy-cell split for a set of
    DG, non-interneuron units, using the two columns in `feature_cols`.

    Parameters
    ----------
    features : pd.DataFrame containing at least the columns in `feature_cols`
        Already restricted by the caller to the DG, non-interneuron population.
    feature_cols : sequence of 2 column names
        The two features to z-score and cluster on (e.g. ["rate", "burst_index"]
        or ["pc1", "log_wake_sleep_ratio"]).
    mossy_feature : str, optional
        Which of `feature_cols` decides cluster identity: the cluster with the
        higher mean of this feature is labeled "mossy cell", the other "granule
        cell". Defaults to `feature_cols[0]` if not given.
    model : {"gmm", "kmeans"}
        Clustering algorithm: "gmm" fits a 2-component GaussianMixture (default),
        "kmeans" fits KMeans(n_clusters=2).
    min_n_units : int
        Minimum number of valid (non-NaN) units required to attempt clustering.
    min_separation : float
        Minimum standardized distance between cluster means (on the composite
        score, the sum of the two z-scored features) required to accept the split;
        below this, everything is labeled "unclassified" rather than forcing an
        arbitrary split.
    random_state : int
        Clustering random seed.

    Returns
    -------
    labels : pd.Series (name="dg_subtype"), values in
        {"granule cell", "mossy cell", "unclassified"}, indexed like `features`.
    diagnostics : dict with keys {model, scaler, cluster_means, separation, used_index}
        (None/empty fields if clustering was skipped).
    """
    if model not in ("gmm", "kmeans"):
        raise ValueError("model must be 'gmm' or 'kmeans', got {!r}".format(model))

    feature_cols = list(feature_cols)
    if mossy_feature is None:
        mossy_feature = feature_cols[0]
    mossy_feature_idx = feature_cols.index(mossy_feature)

    labels = pd.Series("unclassified", index=features.index, name="dg_subtype")
    diagnostics = {"model": None, "scaler": None, "cluster_means": None, "separation": None, "used_index": None}

    valid = features[feature_cols].notna().all(axis=1)
    used_index = features.index[valid]

    if len(used_index) < min_n_units:
        return labels, diagnostics

    X = features.loc[used_index, feature_cols].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    if model == "gmm":
        clusterer = GaussianMixture(n_components=2, random_state=random_state)
    else:
        clusterer = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    cluster_ids = clusterer.fit_predict(Xz)

    composite = Xz.sum(axis=1)
    cluster_means = [composite[cluster_ids == k].mean() for k in range(2)]
    pooled_std = composite.std() + 1e-12
    separation = abs(cluster_means[1] - cluster_means[0]) / pooled_std

    diagnostics.update(
        model=clusterer, scaler=scaler, cluster_means=cluster_means, separation=separation, used_index=used_index
    )

    if separation < min_separation:
        return labels, diagnostics

    mossy_feature_vals = Xz[:, mossy_feature_idx]
    mossy_feature_cluster_means = [mossy_feature_vals[cluster_ids == k].mean() for k in range(2)]
    mossy_cluster = int(np.argmax(mossy_feature_cluster_means))
    assigned = np.where(cluster_ids == mossy_cluster, "mossy cell", "granule cell")
    labels.loc[used_index] = assigned
    return labels, diagnostics