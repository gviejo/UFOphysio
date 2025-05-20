# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-10-02 12:12:53

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
    fig_height = fig_width*golden_mean*0.7         # height in inches
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

name = 'LMN-ADN/A5044/A5044-240401B'
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

aux, timestep = get_memory_map(os.path.join(data.path, data.basename+"_auxiliary.dat"), 3, 20000)
aux = nap.TsdFrame(t=timestep, d=aux)

nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))

channels = data.group_to_channel

structs = ['ADN', 'LMN']

exs = [
    1983.8766,
    1993.87705,
    2023.87835
]

# tcurves = { "ADN":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['adn'], data.position['ry'], 60, data.epochs['wake']),
#             "LMN":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['lmn'], data.position['ry'], 60, data.epochs['wake'])}


# for k in tcurves:
#     tcurves[k] = smoothAngularTuningCurves(tcurves[k], 40, 5)
#     mi = nap.compute_1d_mutual_info(tcurves[k], data.position['ry'], data.epochs['wake'])
#     tcurves[k] = tcurves[k][mi[mi>0.1].dropna().index]


###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(0.95))

outergs = GridSpec(1, 1, figure=fig)


gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, 
    subplot_spec=outergs[0, 0], width_ratios=[0.6, 0.4])


gs_lfp = gridspec.GridSpecFromSubplotSpec(3, 4, 
    subplot_spec=gs0[0, 0], width_ratios=[0.1, 0.5, 0.5, 0.5])

### LFP examples

chs = [0, 6]
yls = ['ADN', 'LMN']

window_size = [0.02, 0.04]

for i, t in enumerate(exs):
    
    tmp = lfp.get(t-window_size[0], t+window_size[1])
    
    for j in range(2):
        subplot(gs_lfp[j,i+1])
        noaxis(gca())
        [plot((tmp[:,c]-k*1000)*2, linewidth=0.5, color=colors[structs[j]]) for k, c in enumerate(channels[chs[j]])]
        xlim(t-window_size[0], t+window_size[1])
        if i == 0:
            ylabel(yls[j], rotation=0, y=0.3, labelpad=15)
        axvline(t, color = COLOR, linewidth=0.1)
    
    subplot(gs_lfp[2,i+1])
    simpleaxis(gca())
    plot(nSS.get(t-window_size[0], t+window_size[1]), color = colors["LMN"], linewidth=1, label="600-2000 Hz")
    xlim(t-window_size[0], t+window_size[1])
    ylim(-3,12)
    gca().spines['bottom'].set_bounds(t+0.03, t+window_size[1])
    xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 ms"])    
    axhline(3.0, linewidth=0.1, color=COLOR, linestyle="--")    
    axvline(t, color = COLOR, linewidth=0.1)
    # ylim(-1, 4)
    if i == 0:
        legend(frameon=False, bbox_to_anchor=(-0.5, -0.75), handlelength=0.0, loc=3)
        ylabel("Power\n(z)", rotation=0, labelpad=20, y=0.1)
    else:
        yticks([])

    # # Accelerometer
    # subplot(gs_lfp[3,i+1])
    # simpleaxis(gca())    
    # plot(aux.get(t-window_size[0], t+window_size[1]))
    # xlim(t-window_size[0], t+window_size[1])
    # gca().spines['bottom'].set_bounds(t+0.03, t+window_size[1])
    # xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 ms"])    
    # axvline(t, color = COLOR, linewidth=0.1)
    # # ylim(-1, 4)
    # if i == 0:
    #     legend(frameon=False, bbox_to_anchor=(-0.5, -0.75), handlelength=0.0, loc=3)
    #     ylabel("Power\n(z)", rotation=0, labelpad=20, y=0.1)
    # else:
    #     yticks([])


### CCS
gs_peth = gridspec.GridSpecFromSubplotSpec(4, 2, 
    subplot_spec=gs0[0, 1], 
    width_ratios=[0.05, 0.3], height_ratios=[0.1, 0.4, 0.2, 0.1],
    wspace = 1, hspace=0.3)

data = cPickle.load(open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/cc_sound.pickle"), 'rb'))

peths = data['peths']
ccs = data['ccs']


# top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('copper', 128)


colors = bottom(np.linspace(0, 1, len(peths)))


subplot(gs_peth[1,1])
simpleaxis(gca())
gca().spines['bottom'].set_visible(False)
count = 0
for i,s in enumerate(peths):
    tmp = peths[s]
    scatter(tmp.index.values, tmp.values+count, s=1, fc=colors[i], ec=None, alpha=0.5)
    count += np.max(tmp) + 50
xticks([])
yticks([])
xlim(-0.05, 0.05)
axvline(0, color = COLOR, linewidth=0.2)
ylabel("Events", rotation=0, labelpad=20)


subplot(gs_peth[2,1])
simpleaxis(gca())
for i in range(len(peths)):
    plot(ccs['sws'].iloc[:,i], color=colors[i], linewidth = 1, alpha=0.5)
axvline(0, color = COLOR, linewidth=0.1)
xlim(-0.05, 0.05)
ylabel("Rate\n(norm.)", rotation=0, labelpad=20, y=0.4)
xlabel("Sound time (ms)")

xticks([-0.05, 0.0, 0.05], [-50, 0, 50])

outergs.update(top=0.95, bottom=0.09, right=0.98, left=0.06)


savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/FENS 2024/figures/fig3.pdf",
    dpi=200,
    facecolor="white",
)
