import os
import pynapple as nap
import scipy
from pynaviz import scope
import numpy as np
import pandas as pd
from ufo_detection import *
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

cc_ufo = nap.compute_eventcorrelogram(nap.TsGroup({0:ufo_tsd}), nap.Ts(opto_ep.start), binsize = 0.001, windowsize = 0.5, norm=True)

perievent = nap.compute_perievent(nap.Ts(ufo_tsd.t), nap.Ts(opto_ep.start), window=(-0.2, 0.3))

dacc_perievent = nap.compute_event_triggered_average(dacc, nap.Ts(opto_ep.start), binsize=0.005, window=(-0.2, 0.3))
dacc_perievent = dacc_perievent[:,0,:]




# spikes = nap.load_file("/Users/gviejo/Downloads/Shan_experiment/B6402-260819/lupin/B6402-260819.npz")
#
# all_ep = pd.read_csv("/Users/gviejo/Downloads/Shan_experiment/B6402-260819/Epoch_TS.csv", index_col=None, header=None)
# all_ep = nap.IntervalSet(all_ep.values[:,0], all_ep.values[:,1])
#
# opto_ep = nap.IntervalSet(opto_ep.start+all_ep.start[1], opto_ep.end+all_ep.start[1])
# ufo_ep = nap.IntervalSet(ufo_ep.start+all_ep.start[1], ufo_ep.end+all_ep.start[1])
# ufo_tsd = nap.Tsd(ufo_tsd.t+all_ep.start[1], ufo_tsd.values)
#
# path2 = "/Users/gviejo/Downloads/Shan_experiment/B6402-260819/B6402_260819_135839"
# analogin_file = os.path.join(path2, "analogin.dat")
#
# fp, timestep = get_memory_map(analogin_file, 1, frequency=20000)
#
# analogin = nap.TsdFrame(t=timestep, d=fp, columns=["tracking"])
#
# csv_file = [f for f in os.listdir(path2) if f.endswith(".csv")][0]
# position = load_optitrack_csv(os.path.join(path2, csv_file))
#
# ttl = analogin[:,0]
# ttl = ttl/ttl.max()
# peaks, _ = scipy.signal.find_peaks(
#     np.diff(ttl.values), height=0.3, distance=int(20000.0 / (120 * 2))
# )
# position = nap.TsdFrame(
#     t=timestep[peaks][0:len(position)],
#     d=position.values,
#     columns = position.columns,
#     time_support = nap.IntervalSet(start=timestep[peaks[0]], end=timestep[peaks[-1]])
# )
#
# ##########################################################
#
# spikes = spikes[spikes.rate > 1.0]
#
# angle = np.deg2rad(position['ry'])+np.pi
#
# tc = nap.compute_tuning_curves(spikes, angle, 120, range=(0, 2*np.pi), epochs = position.time_support.loc[[0]])
# tc.plot(row="unit", col_wrap=5, subplot_kws={"projection": "polar"}, sharey=False)
# SI = nap.compute_mutual_information(tc)
# tokeep = SI[SI["bits/spike"] > 0.2].index.values
#
# spikes = spikes[tokeep]
# tc = tc.sel(unit=tokeep)

# ahv = np.unwrap(angle).smooth(0.05).derivative()
# tc_ahv = nap.compute_tuning_curves(spikes, ahv, 120, range=(-2*np.pi, 2*np.pi), epochs = position.time_support.loc[[0]])
# tc_ahv.plot(row="unit", col_wrap=5, sharey=False)
# decoded, proba_feature = nap.decode_bayes(
#     tuning_curves=tc_ahv,
#     data=spikes,
#     epochs=position.time_support[0],
#     sliding_window_size=4,
#     bin_size=0.1,
# )


# decoded, proba_feature = nap.decode_bayes(
#     tuning_curves=tc,
#     data=spikes,
#     epochs=all_ep[1],
#     sliding_window_size=4,
#     bin_size=0.005,
# )

# ahv_sleep = np.unwrap(decoded)
# ahv_sleep = ahv_sleep.smooth(0.01).derivative()
# # ahv_sleep = np.abs(ahv_sleep)
# # ahv_opto_sleep2 = nap.compute_event_triggered_average(ahv_sleep, nap.Ts(opto_ep.start), binsize=0.02, window=(-0.2, 0.3))
# ahv_opto_sleep = nap.compute_perievent(ahv_sleep, nap.Ts(opto_ep.start), window=(-0.2, 0.3))
# ahv_opto_sleep = pd.DataFrame(
#     index = ahv_opto_sleep.index.values, columns=["mean", "std"], data=
#     np.vstack([ahv_opto_sleep.mean(1), ahv_opto_sleep.std(1)]).T
# )
#
# scope([data, ufo_ep, nSS, opto_ep, spikes])

x_lim = (-0.5, 1.5)
opto_duration = np.mean(opto_ep.end - opto_ep.start)

figure(figsize=(10, 8))
ax = subplot(411)
plot(dacc_perievent)
axvspan(0, opto_duration, color='gray', alpha=0.2)
xlim(-0.1, 0.2)
ylabel("D(Accelerometer)/dt")
ylim(-0.1, 0.1)
subplot(412, sharex=ax)
plot(perievent.to_tsd(), '|')
xlim(-0.1, 0.2)
ylabel("UFOs")
axvspan(0, opto_duration, color='gray', alpha=0.2)
subplot(413, sharex=ax)
plot(cc_ufo)
xlabel("Time from Opto (s)")
ylabel("Normalized Count")
xlim(-0.1, 0.2)
axvspan(0, opto_duration, color='gray', alpha=0.2)
# subplot(414, sharex=ax)
# plot(ahv_opto_sleep['mean'], color='k')
# fill_between(ahv_opto_sleep.index.values, ahv_opto_sleep['mean']-ahv_opto_sleep['std'], ahv_opto_sleep['mean']+ahv_opto_sleep['std'], color='k', alpha=0.2)
# xlabel("Time from Opto (s)")
# ylabel("AHV (rad/s)")
# xlim(-0.1, 0.2)

savefig(os.path.join(path, "opto_ufo_analysis.pdf"), dpi=300)