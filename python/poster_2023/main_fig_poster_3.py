# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-12 17:03:40

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
    fig_height = fig_width*golden_mean*0.8         # height in inches
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

name = 'LMN-PSB/A3019/A3019-220630A'
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

nSS_LMN = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))

channels = data.group_to_channel

channels[0] = list(channels[0])
for c in [45, 47, 43]:
    channels[0].remove(c)

structs = ['PSB', 'LMN']

exs = [
    2830.739
]

tcurves = { "PSB":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['psb'], data.position['ry'], 60, data.epochs['wake']),
            "LMN":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['lmn'], data.position['ry'], 60, data.epochs['wake'])}


for k in tcurves:
    tcurves[k] = smoothAngularTuningCurves(tcurves[k], 40, 5)
    mi = nap.compute_1d_mutual_info(tcurves[k], data.position['ry'], data.epochs['wake'])
    tcurves[k] = tcurves[k][mi[mi>0.1].dropna().index]


###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(0.95))

outergs = GridSpec(1, 3, figure=fig, width_ratios=[0.3, 0.5, 0.4])


# gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, 
#     subplot_spec=outergs[0, 0], height_ratios=[0.2, 0.8])

#####################################
# HISTO 
gs_hs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outergs[0,0], hspace=0.0)


#####################################
# LFP SWS
# gs_lfp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0,1], hspace=0.4, wspace=0.0, height_ratios=[0.4,0.6])

gs_lfp_1 = gridspec.GridSpecFromSubplotSpec(3, 1, 
    subplot_spec=outergs[0,1], hspace=0.1, wspace=0.1,
    height_ratios=[0.9,0.3,0.2]
    )

ep = nap.IntervalSet(exs[0] - 0.015, exs[0] + 0.015)
lws = 0.5
structs = ['PSB', 'LMN']
tmp = lfp.restrict(ep)
tmp = nap.TsdFrame(t=tmp.t, d=tmp.d.astype(float))
names = ['Post\nsub.', 'LMN']

for j, ch in enumerate([0, 2]):
    subplot(gs_lfp_1[j,0])
    noaxis(gca())
    
    [plot((tmp[:,c]-k*1000)*100, linewidth=lws, color=colors[structs[j]]) for k, c in enumerate(channels[ch])]

    xlim(ep[0,0], ep[0,1])

    ylabel(names[j], rotation=0, labelpad=25)

subplot(gs_lfp_1[2,0])
simpleaxis(gca())
plot(nSS_LMN.restrict(ep), color = colors[structs[j]])
xlim(ep[0,0], ep[0,1])
gca().spines['bottom'].set_visible("True")
gca().spines['bottom'].set_bounds(ep.end[0]-0.005, ep.end[0])
xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["5 ms"])
ylabel("Power (z)", rotation=0, labelpad=25, y=0.4)

outergs.update(top=0.95, bottom=0.09, right=0.98, left=0.1)

#####################################
# CROSS_CORR



savefig(
    os.path.expanduser("~") + "/Dropbox/UFOPhysio/figures/poster/fig3.pdf",
    dpi=200,
    facecolor="white",
)
