# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-05 10:38:46

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
from functions.functions import *
import _pickle as cPickle

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
    fig_height = fig_width*golden_mean*0.9         # height in inches
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

name = 'LMN-ADN/A5044/A5044-240402A'
path = os.path.join(data_directory, name)

colors = {"ADN": "#EA9E8D", "LMN": "#8BA6A9", "PSB": "#CACC90", "ctrl":"grey"}

############################################################################################### 
# LOADING DATA
###############################################################################################
data = ntm.load_session(path, 'neurosuite')
data.load_neurosuite_xml(data.path)

fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".eeg"), data.nChannels, 1250)
eeg = nap.TsdFrame(t=timestep, d=fp)

fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".dat"), data.nChannels, 20000)
lfp = nap.TsdFrame(t=timestep, d=fp)


nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))


channels = data.group_to_channel

wak_ex = nap.IntervalSet(7326.459, 7326.759)


ts_wak = [
    #7326.612,
    7330.761,
    7345.842,
    # 7486.110
    7619.862
    ]

position = data.position
ep = position[['x', 'z']].time_support.loc[[0]]
bin_size = 0.01
# lin_velocity = computeLinearVelocity(position[['x', 'z']], ep, bin_size)
# lin_velocity = lin_velocity*100.0

ang_velocity = computeAngularVelocity(position['ry'], ep, bin_size)
# ang_velocity = nap.Tsd(t=position.t, d=np.unwrap(position['ry'])).derivative().bin_average(bin_size).restrict(ep)
ang_velocity = np.abs(ang_velocity)
ang_velocity = ang_velocity/bin_size
ang_velocity = np.rad2deg(ang_velocity)


###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, figure=fig, height_ratios=[0.7, 0.4], hspace=0.5)

gs0 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=outergs[0, 0], width_ratios=[0.05, 0.2, 0.2, 0.2],
    height_ratios=[0.4, 0.2, 0.2]
    )

# LFP
chs = 6
structs = ['LMN']
cut = [0.3, 0.3]
prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2",
            shrinkA=0,shrinkB=0)


for i in range(len(ts_wak)):
    t = ts_wak[i]
    eptmp = nap.IntervalSet(t-cut[0], t+cut[1])
    tmp = lfp.restrict(eptmp)
    subplot(gs0[0,i+1])
    noaxis(gca())
    [plot((tmp[:,c]-k*1000)*1, linewidth=0.25, color=colors["LMN"]) for k, c in enumerate(channels[chs])]
    xlim(t-cut[0], t+cut[1])
    if i == 0:
        ylabel("LMN", rotation=0, y=0.3, labelpad=15)
    else:
        yticks([])    
    annotate("", (t, 1000), (t, 1010), arrowprops=prop)

    


    subplot(gs0[1,i+1])
    simpleaxis(gca())
    tmp = nSS.restrict(eptmp).smooth(0.002)
    plot(tmp, linewidth=1, color = "slategrey")
    xlim(t-cut[0], t+cut[1])
    xticks([])
    if i == 0: 
        ylabel("Power\n(z)", rotation=0, y=0.3, labelpad=15)
    else:
        yticks([])

    gca().spines['bottom'].set_visible(False)

    if i > 0:
        gca().spines['left'].set_visible(False)


    subplot(gs0[2,i+1])
    simpleaxis(gca())
    plot(ang_velocity.restrict(eptmp), color = "darkgreen", linewidth=1)
    xlim(t-cut[0], t+cut[1])
    xticks([])
    gca().spines['bottom'].set_bounds(t+cut[1]-0.1, t+cut[1])
    xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 ms"])
    if i == 0:
        ylabel("Angular\nvelocity\n(deg/s)", rotation=0, y=0.1, labelpad=20)
    else:
        yticks([])

    if i > 0:
        gca().spines['left'].set_visible(False)


    # ax2 = gca().twinx()
    # simpleaxis(ax2)
    # ax2.plot(ang_velocity.get(t-cut[0], t+cut[1]), label="Angular velocity")    
    # if i == 0:
    #     legend()




gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, 
    subplot_spec=outergs[1, 0], wspace = 0.3, hspace=0.5)

data = cPickle.load(open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CORR_UFO_SPEED.pickle"), 'rb'))

eta_linv = data['eta_linv']
eta_angv = data['eta_angv']



gs1_1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[0, 0], wspace = 0.5)
subplot(gs1_1[0,0])
simpleaxis(gca())
tmp = eta_linv - eta_linv.mean()
tmp = tmp / tmp.std()
plot(tmp, color='grey', alpha=0.2, linewidth=0.1)
plot(tmp.mean(1), linewidth=1, color = "midnightblue")
# plot(eta_linv.mean(1))
title("Linear velocity")
axvline(0.0, linewidth=0.5, color = COLOR)
ylim(-3, 3)
ylabel("z")
xlabel("UFO time (s)")

subplot(gs1_1[0,1])
simpleaxis(gca())
tmp = eta_angv - eta_angv.mean()
tmp = tmp / tmp.std()
plot(tmp, color='grey', alpha=0.2, linewidth=0.1)
plot(tmp.mean(1), linewidth=1, color = "darkgreen")
# plot(eta_angv.mean(1))
title("Angular velocity")
axvline(0.0, linewidth=0.5, color = COLOR)
ylim(-3, 3)
ylabel("z")
xlabel("UFO time (s)")


#########

gs1_2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[0, 1], hspace=0.5)

trans_start = data['trans_start']
trans_stop = data['trans_stop']
peth_start = data['peth_start']
peth_stop = data['peth_stop']


subplot(gs1_2[0,0])
gca().spines['bottom'].set_visible(False)
simpleaxis(gca())
offset = 0
for i, s in enumerate(peth_start.keys()):
    plot(peth_start[s].loc[-0.4:0.4].index.values, peth_start[s].loc[-0.4:0.4].values + offset, 'o', markersize=0.5, markeredgewidth=0, markerfacecolor='black')
    offset += len(peth_start[s])+5
xlim(-0.4, 0.4)
yticks([offset], [len(peth_start)])
xticks([])
ylabel("Sessions", rotation=0, y=0.3, labelpad=15)
title("pause -> turn")

subplot(gs1_2[1,0])
simpleaxis(gca())
tmp = trans_start.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=2)
# plot(tmp.loc[-0.4:0.4], alpha = 0.5, linewidth = 1)
plot(tmp.loc[-0.4:0.4].mean(1), color='grey', linewidth = 1)
m = tmp.loc[-0.4:0.4].mean(1)
s = tmp.loc[-0.4:0.4].std(1).values
fill_between(m.index.values, m.values - s, m.values + s, alpha = 0.5, color = 'grey', linewidth=0)
axvline(0.0, linewidth=0.5, color = COLOR)
maxv = np.max(m.values + s)
ylim(0, maxv+1)
legend(frameon=False)
xlabel("UFO time (s)")
xlim(-0.4, 0.4)

subplot(gs1_2[0,1])
gca().spines['bottom'].set_visible(False)
simpleaxis(gca())
offset = 0
for i, s in enumerate(peth_start.keys()):
    plot(peth_stop[s].loc[-0.4:0.4].index.values, peth_stop[s].loc[-0.4:0.4].values + offset, '.', markersize=0.5, markeredgewidth=0, markerfacecolor='black')
    offset += len(peth_stop[s])+1
xlim(-0.4, 0.4)
yticks([offset], [len(peth_start)])
xticks([])
title("turn -> pause")

subplot(gs1_2[1,1])
simpleaxis(gca())
tmp = trans_stop.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=2)
# plot(tmp.loc[-0.4:0.4], alpha = 0.5, linewidth = 1)
plot(tmp.loc[-0.4:0.4].mean(1), color='grey', linewidth = 1)
m = tmp.loc[-0.4:0.4].mean(1)
s = tmp.loc[-0.4:0.4].std(1).values
fill_between(m.index.values, m.values - s, m.values + s, alpha = 0.5, color = 'grey', linewidth=0)
axvline(0.0, linewidth=0.5, color=COLOR)
legend(frameon=False)
ylim(0, maxv+1)
xlabel("UFO time (s)")

outergs.update(top=0.98, bottom=0.1, right=0.98, left=0.05)


savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/SFN 2025/figures/fig6_head_mov.pdf",
    dpi=200,
    facecolor="white",
)