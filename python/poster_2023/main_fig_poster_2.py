# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-25 17:30:32

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
    fig_height = fig_width*golden_mean*0.5         # height in inches
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

name = 'LMN-ADN/A5002/A5002-200305A'
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

channels = data.group_to_channel

structs = ['LMN']

t_ex = 144.590

ep = nap.IntervalSet(t_ex-0.15, t_ex+0.15)

exs = [144.525, 144.650]

###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(1))

outergs = GridSpec(1, 3, figure=fig, width_ratios=[0.3, 0.5, 0.2])


### HISTO
gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, 
    subplot_spec=outergs[0, 0])


file = os.path.join(os.path.expanduser("~"),"Dropbox/UFOPhysio/figures/poster/MB_histo_graph.png")

subplot(gs0[0,0])
noaxis(gca())
img = mpimg.imread(file)
imshow(img, aspect="equal")
# title(structs[i])
xticks([])
yticks([])



### LFP examples
gs1 = gridspec.GridSpecFromSubplotSpec(4, 1, 
    subplot_spec=outergs[0, 1])


structs = ['LMN', 'ctrl', 'ctrl', 'ctrl']
tmp = lfp.restrict(ep)

for i, ch in enumerate([2,3,4,5]):

    subplot(gs1[i,0])
    noaxis(gca())

    for k, c in enumerate(channels[ch]):    
        plot((tmp[:,c]-k*1000)*1, 
            linewidth=0.3, color=colors[structs[i]], alpha=0.5
            )

    xlim(ep[0,0], ep[0,1])

    if i == 0:
        prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2",
                    shrinkA=0,shrinkB=0)
        for k in range(len(exs)):
            annotate("", (exs[k], 0), (exs[k], 0.5), arrowprops=prop)

    if i == 3:
        gca().spines['bottom'].set_visible("True")
        gca().spines['bottom'].set_bounds(ep.end[0]-0.05, ep.end[0])
        xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["50 ms"])


#### Mean

data2 = cPickle.load(open("/mnt/home/gviejo/Dropbox/UFOPhysio/figures/poster/mb_control.pickle", 'rb'))

gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outergs[0, 2])

bins = np.geomspace(50, 60000, 100)

pwr_ufo = np.array([np.histogram(data2['ufos'][s].values, bins)[0] for s in data2['ufos'].keys()]).T
pwr_ctr = np.array([np.histogram(data2['ctrl'][s].values, bins)[0] for s in data2['ctrl'].keys()]).T
pwr_ufo = pwr_ufo/pwr_ufo.mean(0)
pwr_ctr = pwr_ctr/pwr_ctr.mean(0)


subplot(gs1[1,0])
simpleaxis(gca())
semilogx(bins[0:-1], pwr_ufo, color = colors['LMN'], alpha = 1, linewidth=2)
semilogx(bins[0:-1], pwr_ctr, color = colors['ctrl'], alpha = 1, linewidth=2)


xlabel("Power (uv^2)")

outergs.update(top=0.95, bottom=0.09, right=0.98, left=0.06)


savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/FENS 2024/figures/fig2.pdf",
    dpi=200,
    facecolor="white",
)
