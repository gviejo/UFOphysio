import os
import pynapple as nap
import scipy
from pynaviz import scope
import numpy as np
import pandas as pd
from detection import *
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages

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


def process_session(path):
    """Load one session and compute the opto-triggered AHV / UFO responses.

    Args:
        path (str): Path to the session folder
    """
    basename = os.path.basename(path)

    data = nap.EphysReader(path, format="neurosuite")

    lfp = data[basename + ".dat"]

    analogin_file = os.path.join(path, "analogin.dat")

    fp, timestep = get_memory_map(analogin_file, 2, frequency=20000)

    analogin = nap.TsdFrame(t=timestep, d=fp, columns=["opto", "tracking"])

    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    position = load_optitrack_csv(os.path.join(path, csv_file))

    ttl = analogin[:,1]
    ttl = ttl/ttl.max()
    peaks, _ = scipy.signal.find_peaks(
        np.diff(ttl.values), height=0.3, distance=int(20000.0 / (120 * 2))
    )
    position = nap.TsdFrame(
        t=timestep[peaks][0:len(position)],
        d=position.values,
        columns = position.columns,
        time_support = nap.IntervalSet(start=timestep[peaks[0]], end=timestep[peaks[-1]])
    )

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

    position = nap.TsdFrame(t=position.index.values, d=position.values, columns = position.columns)

    ahv = np.unwrap(position['ry']).derivative()
    ahv = ahv.interpolate(nap.Ts(np.arange(ahv.time_support.start[0], ahv.time_support.end[0], 1/200)))

    # ahv_opto = nap.compute_event_triggered_average(ahv, nap.Ts(opto_ep.start), binsize=0.05, window=(-0.5, 1.5))
    ahv_opto = nap.compute_perievent(ahv, nap.Ts(opto_ep.start), window=(-0.5, 1.5))
    ahv_opto = pd.DataFrame(
        index = ahv_opto.index.values, columns=["mean", "std"], data=
        np.vstack([ahv_opto.mean(1), ahv_opto.std(1)]).T
    )

    sign_channels = lfp.anatomy[lfp.group == 0].values
    ctrl_channels = lfp.anatomy[lfp.group == 2].values

    ufo_ep, ufo_tsd, nSS = detect_ufos_v2(lfp.values, sign_channels, ctrl_channels, lfp.t)
    ufo_ep.nss = ufo_tsd.values
    ufo_ep = ufo_ep[ufo_ep.nss > 5]
    ufo_tsd = ufo_tsd[ufo_tsd > 5]

    cc_ufo = nap.compute_eventcorrelogram(nap.TsGroup({0:ufo_tsd}), nap.Ts(opto_ep.start), binsize = 0.001, windowsize = 1, norm=True)

    perievent = nap.compute_perievent(nap.Ts(ufo_tsd.t), nap.Ts(opto_ep.start), window=(-0.5, 1.5))

    scope([data, position['ry'], ufo_ep, nSS, opto_ep])

    return ahv_opto, perievent, cc_ufo, opto_ep


def make_figure(basename, ahv_opto, perievent, cc_ufo, opto_ep):
    """Build the summary figure of one session.

    Args:
        basename (str): Session name, used as figure title
    """
    x_lim = (-0.5, 1.5)
    opto_duration = np.mean(opto_ep.end - opto_ep.start)

    fig = figure(figsize=(10, 8))
    suptitle(basename)
    ax = subplot(311)
    plot(ahv_opto['mean'], color='k')
    fill_between(ahv_opto.index.values, ahv_opto['mean']-ahv_opto['std'], ahv_opto['mean']+ahv_opto['std'], color='k', alpha=0.2)
    xlim(*x_lim)
    ylabel("AHV (rad/s)")
    axvspan(0, opto_duration, color='gray', alpha=0.2)
    subplot(312, sharex=ax)
    plot(perievent.to_tsd(), '|')
    xlim(*x_lim)
    ylabel("UFOs")
    axvspan(0, opto_duration, color='gray', alpha=0.2)
    subplot(313, sharex=ax)
    plot(cc_ufo)
    xlabel("Time from Opto (s)")
    ylabel("Normalized Count")
    xlim(*x_lim)
    axvspan(0, opto_duration, color='gray', alpha=0.2)
    tight_layout()
    return fig

def make_summary_figure(all_ahv_opto, all_opto_duration):
    """Overlay the opto-triggered AHV of every session on a single panel.

    Args:
        all_ahv_opto (dict): Session name -> ahv_opto dataframe
        all_opto_duration (dict): Session name -> mean opto duration
    """
    x_lim = (-0.5, 1.5)

    fig = figure(figsize=(10, 8))
    suptitle("All sessions")
    colors = cm.viridis(np.linspace(0, 0.8, len(all_ahv_opto)))
    ax = subplot(111)
    for c, (basename, ahv_opto) in zip(colors, all_ahv_opto.items()):
        plot(ahv_opto['mean'], color=c, label=basename)
        fill_between(ahv_opto.index.values, ahv_opto['mean']-ahv_opto['std'], ahv_opto['mean']+ahv_opto['std'], color=c, alpha=0.1)
    xlim(*x_lim)
    ylabel("AHV (rad/s)")
    legend(fontsize="small")
    tight_layout()
    return fig


paths = [
    # "/Users/gviejo/Downloads/Shan_experiment/B6402_260817_121531",
    "/Users/gviejo/Downloads/Shan_experiment/B6402_260817_133500",
    "/Users/gviejo/Downloads/Shan_experiment/B6402_260819_143724"
]

pdf_file = os.path.join(
    os.path.dirname(paths[0]), "opto_ufo_analysis.pdf"
)

all_ahv_opto = {}
all_opto_duration = {}

with PdfPages(pdf_file) as pdf:
    for path in paths:
        basename = os.path.basename(path)
        print("Processing", basename)
        ahv_opto, perievent, cc_ufo, opto_ep = process_session(path)
        fig = make_figure(basename, ahv_opto, perievent, cc_ufo, opto_ep)
        pdf.savefig(fig, dpi=300)
        close(fig)
        all_ahv_opto[basename] = ahv_opto
        all_opto_duration[basename] = np.mean(opto_ep.end - opto_ep.start)

    fig = make_summary_figure(all_ahv_opto, all_opto_duration)
    pdf.savefig(fig, dpi=300)
    close(fig)

print("Saved", pdf_file)
