# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-15 17:51:14

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
#matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg
sys.path.append("../")
from functions import *
from ufo_detection import loadRipples, loadUFOs
import _pickle as cPickle
from scipy.ndimage import gaussian_filter

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
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    duration = n_samples/frequency
    interval = 1/frequency
    f.close()
    fp = np.memmap(filepath, np.int16, 'r', shape = (n_samples, n_channels))        
    timestep = np.arange(0, n_samples)/frequency

    return fp, timestep

def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    # fig_width = 5
    fig_height = fig_width*golden_mean*1         # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)

def compute_csd(lfp, dz=0.1, sigma=1.0):
    """
    lfp: array (n_channels, n_time)
    dz: spacing in mm
    sigma: conductivity (relative or absolute)
    """
    csd = -(sigma * (lfp[2:] - 2*lfp[1:-1] + lfp[:-2])) / (dz**2)
    return csd


fontsize = 8

COLOR = (0.25, 0.25, 0.25)

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = fontsize
rcParams['text.color'] = COLOR
rcParams['axes.labelcolor'] = COLOR
rcParams['axes.labelsize'] = fontsize
rcParams['axes.labelpad'] = 1
#rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize
rcParams['figure.titlesize'] = fontsize
rcParams['xtick.major.size'] = 1.3
rcParams['ytick.major.size'] = 1.3
rcParams['xtick.major.width'] = 0.4
rcParams['ytick.major.width'] = 0.4
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1
rcParams['axes.linewidth'] = 0.4
rcParams['axes.edgecolor'] = COLOR
rcParams['axes.axisbelow'] = True
rcParams['xtick.color'] = COLOR
rcParams['ytick.color'] = COLOR

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



colors = {"ADN": "#EA9E8D", "LMN": "#8BA6A9", "PSB": "#CACC90", "ctrl":"grey", "HPC":"#C9D0AC", "CA1":"#C9D0AC"}


###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(0.9))

outergs = GridSpec(2, 1, figure=fig, height_ratios=[0.4, 0.25], hspace=0.2)

###############################################################################################
# LOADING DATA HPC - ADN
###############################################################################################
exs = {
    "ADN-HPC/B5100/B5101/B5101-250501": [3954.061, 3966.706],
    "ADN-HPC/B5100/B5101/B5101-250430": [
        14228.653,
        # 14235.839,
        14254.017,
        14284.016,
        # 15114.056
    ]
}

name = 'ADN-HPC/B5100/B5101/B5101-250430'
path = os.path.join(data_directory, name)
data = ntm.load_session(path, 'neurosuite')
data.load_neurosuite_xml(data.path)
fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".eeg"), data.nChannels, 1250)
eeg = nap.TsdFrame(t=timestep, d=fp)
fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".dat"), data.nChannels, 20000)
lfp = nap.TsdFrame(t=timestep, d=fp)
# nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
channels = data.group_to_channel
nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
ufo_ep, ufo_ts = loadUFOs(path)
chs = [0, 3]
yls = ['ADN']
lws = [0.5, 0.5, 0.5, 0.5]
alphas = [1, 0.5, 1, 0.5]
labels = ['HD wave', 'CA1 SWRs']
clrs = ['LMN', 'CA1']
window_size = [0.03, 0.07]

gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[0, 0], width_ratios=[0.6, 0.2], wspace=0.5)

##############
# LFP example
##############
gs1 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs_top[0, 0],
                                       height_ratios=[0.3, 0.2, 0.06], hspace=0.15,
                                       width_ratios=[0.07, 0.3, 0.3, 0.3]
                                       )

for i, t in enumerate(exs[name]):

    tmp = lfp.get(t - window_size[0], t + window_size[1])
    tmp = nap.TsdFrame(t=tmp.index.values, d=tmp.values.astype(np.float32))

    tmp2 = eeg.get(t - window_size[0], t + window_size[1])
    tmp2 = nap.TsdFrame(t=tmp2.index.values, d=tmp2.values.astype(np.float32))

    # HPC
    subplot(gs1[0, i+1])
    noaxis(gca())

    ch_to_use = channels[0][channels[0] != 40][15:]

    csd = compute_csd(tmp2[:, ch_to_use].values.T)
    csd = gaussian_filter(csd, sigma=2)



    im = imshow(csd, aspect='auto', origin='upper', cmap='bwr',
           extent=[t - window_size[0], t + window_size[1], -3000 * (len(ch_to_use)), 0],
            vmin=-np.max(np.abs(csd)), vmax=np.max(np.abs(csd))
           )

    if i == 2:
        cbar = colorbar(im,fraction=0.02, pad=0.04)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.set_yticks([-np.max(np.abs(csd)), np.max(csd)])
        cbar.ax.set_yticklabels(["Sink", "Source"])

    for k, c in enumerate(ch_to_use):
        plot((tmp2[:, c] - k * 3000) * 1, linewidth=0.1, color="darkgrey")



    if i == 0:
        ylabel("HPC", rotation=0, y=0.3, labelpad=12)

    # ADN
    subplot(gs1[1, i+1])
    noaxis(gca())

    for k, c in enumerate(channels[chs[1]][2:]):
        plot((tmp[:, c] - k * 1000) * 2, linewidth=0.5, color=colors["ADN"])

    if i == 0:
        ylabel("ADN", rotation=0, y=0.3, labelpad=12)

    # NSS
    subplot(gs1[2, i+1])
    simpleaxis(gca())
    plot(nSS.get(t - window_size[0], t + window_size[1]), color=colors["ADN"], linewidth=1, label="600-2000 Hz")
    xlim(t - window_size[0], t + window_size[1])
    ylim(-3, 10)
    gca().spines['bottom'].set_bounds(t + window_size[1] - 0.03, t + window_size[1])
    xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds()) / 2, ["30 ms"])
    axhline(3.0, linewidth=0.1, color=COLOR, linestyle="--")
    if i == 0:
        legend(frameon=False, bbox_to_anchor=(-0.5, -1.3), handlelength=0.0, loc=3)
        ylabel("Power\n(z)", rotation=0, labelpad=20, y=0.1)
    else:
        yticks([])


# ######################################
# Cross corr

data = cPickle.load(open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CC_UFO_DS.pickle"), 'rb'))

gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_top[0, 1], hspace=0.3,
                                       height_ratios=[0.2, 0.5, 0.2])

# # Raster
# subplot(gs2[1,0])
# simpleaxis(gca())
# for i, s in enumerate(["ADN-HPC/B5100/B5101/B5101-250430"]):
#     plot(data["perievent"][s], '|', markersize=1, color='gray')
#


# CC
subplot(gs2[1,0])
simpleaxis(gca())
plot(data["short"], color='gray', alpha=0.25, linewidth=0.5)
plot(data["short"].mean(1), linewidth = 1, color = "gray")
xlabel("UFO time (s)", labelpad=0)
ylabel("Rate DS\n(norm.)", labelpad=10, y =0.4)
title("Dentate spikes - UFO\ncross-correlation")
axvline(0.0, color=COLOR, linewidth=0.5, linestyle="-")




###############################################################################################
# LOADING DATA HPC - LMN
###############################################################################################
name = 'LMN-ADN/A5022/A5022-210527A'
path = os.path.join(data_directory, name)
data = ntm.load_session(path, 'neurosuite')
data.load_neurosuite_xml(data.path)
fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".eeg"), data.nChannels, 1250)
eeg = nap.TsdFrame(t=timestep, d=fp)
fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".dat"), data.nChannels, 20000)
lfp = nap.TsdFrame(t=timestep, d=fp)
# nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
channels = data.group_to_channel
ufo_ep, ufo_ts = loadUFOs(path)
rip_ep, rip_ts = loadRipples(path)
structs = ['CA1', 'LMN']
chs = [4]
yls = ['LMN']
structs = ['LMN']
lws = [0.5, 0.5, 0.5, 0.5]
alphas = [1, 0.5, 1, 0.5]
ts_ex = 1913.856
# ts_ex = 15.816
labels = ['UFO', 'CA1 SWRs']
clrs = ['LMN', 'CA1']


gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[1, 0], width_ratios=[0.6, 0.2], wspace=0.5)

gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0, 0], height_ratios=[0.5, 0.3], hspace=0.2)

gs0_0 = gridspec.GridSpecFromSubplotSpec(2, 2,
                                         subplot_spec=gs0[0, 0],
                                         width_ratios=[0.1, 0.9],
                                         height_ratios=[0.5, 0.5], hspace=0.2, wspace=0.1)


# ######################################
# HISTOGRAM
subplot(gs0_0[0,1])
noaxis(gca())
# gca().spines['bottom'].set_visible(False)

ep = nap.IntervalSet(ts_ex-80, ts_ex+80)

ufo_ts = ufo_ts.restrict(ep)
rip_ts = rip_ts.restrict(ep)

plot(rip_ts.t, np.random.randn(len(rip_ts))+10, '|', color = colors["CA1"], markersize=1)
plot(ufo_ts.t, np.random.randn(len(ufo_ts)), '|', color = colors['LMN'], markersize=1)

yticks([0, 10], labels)
xlim(ep[0,0], ep[0,1])
ylim(-10, 20)

xticks([])
axvspan(ts_ex-1, ts_ex+1, color = 'grey', alpha=0.25, linewidth=0)

#
subplot(gs0_0[1,1])
simpleaxis(gca())
gca().spines['bottom'].set_visible(False)
binsize = 0.5
for i, ts in enumerate([ufo_ts, rip_ts]):
    tmp = ts.count(binsize)
    bar(tmp.t, tmp.d, binsize, color = colors[clrs[i]], label = labels[i], edgecolor=colors[clrs[i]])
xlim(ep[0,0], ep[0,1])
ylabel("Count", rotation=0, labelpad=20, y=0.5)
axvspan(ts_ex-1, ts_ex+1, color = 'grey', alpha=0.25, linewidth=0)
gca().spines['bottom'].set_bounds(ep[0,1]-10, ep[0,1])
xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 s"])
ax1 = gca()

# ######################################
# LFP
# gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1, 0], width_ratios=[0.4, 0.2], wspace=0.5)

gs_ex = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1,0])#, width_ratios=[0.1, 0.6, 0.1])#, hspace=0.0, wspace=0.0)
tmp = lfp.get(ts_ex-1, ts_ex+1)

prop = dict(arrowstyle="-|>,head_width=0.15,head_length=0.3",
            shrinkA=0,shrinkB=0)


# CA1
subplot(gs_ex[0,0])
noaxis(gca())
plot((tmp[:,3]), linewidth=0.5, color=colors['CA1'])
ax2 = gca()
xlim(ts_ex-1, ts_ex+1)
ylabel("CA1", rotation=0, y=0.3, labelpad=12)
for t in rip_ts.restrict(nap.IntervalSet(ts_ex-1, ts_ex+1)).t:
    annotate("", (t, gca().get_ylim()[0]), (t, gca().get_ylim()[0]-100), arrowprops=prop)

#LMN
subplot(gs_ex[1,0])
noaxis(gca())
[plot((tmp[:,c]-k*1000)*2, linewidth=0.5, color=colors["LMN"]) for k, c in enumerate(channels[chs[0]][::2])]
ylabel("LMN", rotation=0, y=0.3, labelpad=12)
xlim(ts_ex-1, ts_ex+1)
gca().spines['bottom'].set_visible(True)
gca().spines['bottom'].set_bounds(gca().get_xlim()[1]-0.5, gca().get_xlim()[1])
xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["500 ms"])
annotate("", (ts_ex, gca().get_ylim()[1]), (ts_ex, gca().get_ylim()[1]+100), arrowprops=prop)


# from matplotlib.patches import ConnectionPatch
#
# xy = (ts_ex-1, 0)
# con = ConnectionPatch(xyA=(0, 1), xyB=xy, coordsA="axes fraction", coordsB="data",
#                       axesA=ax2, axesB=ax1, color=COLOR, linewidth=0.01)
# ax2.add_artist(con)
#
# xy = (ts_ex+1, 0)
# con = ConnectionPatch(xyA=(1, 1), xyB=xy, coordsA="axes fraction", coordsB="data",
#                       axesA=ax2, axesB=ax1, color=COLOR, linewidth=0.01)
# ax2.add_artist(con)


# ######################################
# # CC
gs_cc = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_bottom[0, 1], hspace=0.5, height_ratios=[0.5, 0.5, 0.03])

data = cPickle.load(open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CC_UFO_SWR.pickle"), 'rb'))

cc_long = data["long"]
cc_short = data["short"]





subplot(gs_cc[0,0])
simpleaxis(gca())

cc_long = cc_long.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=2)

plot(cc_long, alpha = 0.25, linewidth=0.5, color = "grey")
plot(cc_long.mean(1), linewidth = 1, color = 'gray', label = "SWR")
# xlabel("HD wave\ntime (s)")
ylabel("Rate SWRs\n(norm.)", labelpad=10, y =0.0)
title("SWRs - UFO\ncross-correlation")

subplot(gs_cc[1,0])
simpleaxis(gca())

cc_short = cc_short.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3)

plot(cc_short, alpha = 0.25, linewidth=0.5, color = "grey")
plot(cc_short.mean(1), linewidth = 1, color = 'gray', label = "SWR")
xlabel("UFO time (s)", labelpad=0)
# ylabel("Rate SWRs\n(norm.)", labelpad=10, y =0.4)



outergs.update(top=0.99, bottom=0.04, right=0.98, left=0.05)


savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/SFN 2025/figures/fig4_swr_ds_corr.pdf",
    dpi=200,
    facecolor="white",
)
