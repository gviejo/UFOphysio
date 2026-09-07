import os, sys
import pynapple as nap
import scipy
from pynaviz import scope
import numpy as np
import pandas as pd
from detection import *
from matplotlib.pyplot import *

def get_memory_map(filepath, nChannels, frequency=20000):
    """Summary

    Args:
        filepath (TYPE): Description
        nChannels (TYPE): Description
        frequency (int, optional): Description
    """
    n_channels = int(nChannels)
    f = open(filepath, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2
    n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
    duration = n_samples / frequency
    interval = 1 / frequency
    f.close()
    fp = np.memmap(filepath, np.uint16, 'r', shape=(n_samples, n_channels))
    timestep = np.arange(0, n_samples) / frequency

    return fp, timestep

def load_optitrack_csv(csv_file):
    position = pd.read_csv(csv_file, header=[4, 5], index_col=1)
    if 1 in position.columns:
        position = position.drop(labels=1, axis=1)
    position = position[~position.index.duplicated(keep="first")]
    order = []
    cols = []
    for n in position.columns:
        if n[0] == "Rotation":
            order.append("r" + n[1].lower())
            cols.append(n)
        elif n[0] == "Position":
            order.append(n[1].lower())
            cols.append(n)
    if len(order) == 0:
        raise RuntimeError(
            "Unknow tracking format for csv file {}".format(csv_file)
        )
    position = position[cols]
    position.columns = order
    return position


paths = ["/Users/gviejo/Downloads/Shan_experiment/B6402_260817_153414",
         "/Users/gviejo/Downloads/Shan_experiment/B6402-260819/B6402_260819_170352"]

# path = "/Users/gviejo/Downloads/Shan_experiment/B6402_260817_153414"
# path = "/Users/gviejo/Downloads/Shan_experiment/B6402-260819/B6402_260819_170352"
path = "/Users/gviejo/Downloads/Shan_experiment/B6402_260824_110820"

basename = os.path.basename(path)

data = nap.EphysReader(path, format="neurosuite")

lfp = data[basename + ".dat"]

# Sleep scoring
channels = [chs[0] for gr, chs in lfp.metadata.groupby("group").groups.items() if lfp.metadata.loc[chs[0]] is not False]
eps = 1e-10
tmp = lfp.loc[channels].decimate(80)
theta = nap.apply_bandpass_filter(tmp, (6, 12), fs=250)
delta = nap.apply_bandpass_filter(tmp, (1, 4), fs=250)
theta_amp = nap.compute_hilbert_envelope(theta)
delta_amp = nap.compute_hilbert_envelope(delta)

theta_amp = theta_amp.mean(1)
delta_amp = delta_amp.mean(1)

theta_amp = theta_amp.smooth(0.02)
delta_amp = delta_amp.smooth(0.02)

ratio = 2 * np.log10(
    (theta_amp.values + eps) /
    (delta_amp.values + eps)
)

ratio = nap.Tsd(t=tmp.index.values, d=ratio)
ratio = ratio.smooth(0.1)

rem_ep = ratio.threshold(0.5).time_support.merge_close_intervals(1.0).drop_short_intervals(1.0)
sws_ep = ratio.threshold(-1.0, method="below").time_support.merge_close_intervals(1.0).drop_short_intervals(1.0)


analogin_file = os.path.join(path, "analogin.dat")

fp, timestep = get_memory_map(analogin_file, 1, frequency=20000)

analogin = nap.TsdFrame(t=timestep, d=fp, columns=["opto"])

auxiliary_file = os.path.join(path, "auxiliary.dat")

fp, timestep = get_memory_map(auxiliary_file, 3, frequency=20000)
accelerometer = nap.TsdFrame(t=timestep, d=fp, columns=["x", "y", "z"])
accelerometer = accelerometer.decimate(16)
dacc = accelerometer.derivative()
dacc = dacc - dacc.mean(0)
dacc = dacc / dacc.std(0)

ttl2 = analogin[:,0]
ttl2 = ttl2/ttl2.max()
peaks1, _ = scipy.signal.find_peaks(
    np.diff(ttl2.values), height=0.5
)
peaks2, _ = scipy.signal.find_peaks(
    -np.diff(ttl2.values), height=0.5
)
opto_ep = nap.IntervalSet(
    start=timestep[peaks1],
    end=timestep[peaks2])

sign_channels = lfp.anatomy[lfp.group == 0].values
ctrl_channels = lfp.anatomy[lfp.group == 2].values

ufo_ep, ufo_tsd, nSS = detect_ufos_v2(lfp.values, sign_channels, ctrl_channels, lfp.t)
ufo_ep.nss = ufo_tsd.values
ufo_ep = ufo_ep[ufo_ep.nss > 5]
ufo_tsd = ufo_tsd[ufo_tsd > 5]


cc_ufo = {}
perievent = {}
dacc_perievent = {}

for ep, ep_name in zip([rem_ep, sws_ep], ["REM", "SWS"]):
    opto_ts = nap.Ts(ep.intersect(opto_ep).start)
    ufo_tsg = nap.TsGroup({0:ufo_tsd}, time_support=ep)
    cc_ufo[ep_name] = nap.compute_eventcorrelogram(ufo_tsg, opto_ts, binsize = 0.001, windowsize = 0.5, norm=False)
    perievent[ep_name] = nap.compute_perievent(nap.Ts(ufo_tsd.t), opto_ts, window=(-0.2, 0.3))
    dacc_perievent[ep_name] = nap.compute_event_triggered_average(dacc, opto_ts, binsize=0.005, window=(-0.2, 0.3))
    dacc_perievent[ep_name] = dacc_perievent[ep_name][:,0,:]


# scope([data, ufo_ep, nSS, opto_ep, rem_ep, sws_ep])
scope({"data": data, "ufo_ep": ufo_ep, "nSS": nSS, "opto_ep": opto_ep, "rem_ep": rem_ep, "sws_ep": sws_ep, "ratio": ratio})

x_lim = (-0.5, 1.5)
opto_duration = np.mean(opto_ep.end - opto_ep.start)

figure(figsize=(10, 8))
ax = subplot(411)
for ep_name in ["REM", "SWS"]:
    plot(dacc_perievent[ep_name].mean(1), label=ep_name)
axvspan(0, opto_duration, color='gray', alpha=0.2)
legend()
xlim(-0.1, 0.2)
ylabel("D(Accelerometer)/dt")
ylim(-0.1, 0.1)
subplot(412, sharex=ax)
plot(perievent["REM"].to_tsd(), '|', color='blue')
xlim(-0.1, 0.2)
ylim(0, len(perievent["REM"]))
ylabel("UFOs")
axvspan(0, opto_duration, color='gray', alpha=0.2)
subplot(413, sharex=ax)
plot(perievent["SWS"].to_tsd(), '|', color='orange')
xlim(-0.1, 0.2)
ylim(0, len(perievent["SWS"]))
ylabel("UFOs")
axvspan(0, opto_duration, color='gray', alpha=0.2)
subplot(414, sharex=ax)
for ep_name in ["REM", "SWS"]:
    plot(cc_ufo[ep_name]/len(perievent[ep_name]), label=ep_name)
legend()
xlabel("Time from Opto (s)")
ylabel("Normalized Count")
xlim(-0.1, 0.2)
axvspan(0, opto_duration, color='gray', alpha=0.2)


savefig(os.path.join(path, "opto_ufo_analysis.png"), dpi=300)