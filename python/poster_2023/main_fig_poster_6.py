# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-29 11:07:05

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

name = 'LMN-ADN/A5022/A5022-210527A'
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





structs = ['CA1', 'LMN']

###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(0.9))

outergs = GridSpec(2, 1, figure=fig, height_ratios=[0.5, 0.1], hspace=0.2)

gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outergs[0, 0], height_ratios=[0.2, 0.4])


chs = [4]
yls = ['LMN']
structs = ['LMN']
lws = [0.5, 0.5, 0.5, 0.5]
alphas = [1, 0.5, 1, 0.5]


ts_ex = 1913.856
# ts_ex = 15.816


# ######################################
# PWR
pwrs = nap.load_file(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/pwr_ufo_ca1.npz"))

subplot(gs0[0,0])
simpleaxis(gca())
gca().spines['bottom'].set_visible(False)


ep = nap.IntervalSet(ts_ex-10, ts_ex+10)

for i, name in enumerate(['ufo', 'ca1']):
    tmp = pwrs[name].smooth(0.5).restrict(ep)
    tmp = tmp - np.mean(tmp)
    tmp = tmp / np.std(tmp)
    plot(tmp, label=name)
legend()

axvspan(ts_ex-1.0, ts_ex+1.0, alpha=0.5, linewidth = 0)


gs_ex = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1,0])#, hspace=0.0, wspace=0.0)
tmp = lfp.get(ts_ex-1, ts_ex+1)
# CA1

subplot(gs_ex[0,0])
noaxis(gca())
plot((tmp[:,3]), linewidth=0.5, color="darkgreen")

#LMN
subplot(gs_ex[1,0])
noaxis(gca())
[plot((tmp[:,c]-k*1000)*2, linewidth=0.5, color=colors["LMN"]) for k, c in enumerate(channels[chs[0]][::2])]
# xlim(t-0.02, t+0.04)
# ylabel(yls[j], rotation=0, y=0.3, labelpad=15)
    

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outergs[1, 0])#, width_ratios=[0.2, 0.8])



# ######################################
# # CC
data = cPickle.load(open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CC_UFO_SWR.pickle"), 'rb'))

cc_long = data["long"]
cc_short = data["short"]



cc_long = cc_long.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)


subplot(gs1[0,1])
simpleaxis(gca())

plot(cc_long, alpha = 0.25, linewidth=1, color = "grey")
plot(cc_long.mean(1), linewidth = 4, color = 'red', label = "SWR")
xlabel("ufo (s)")



outergs.update(top=0.98, bottom=0.09, right=0.98, left=0.05)


savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/FENS 2024/figures/fig6.pdf",
    dpi=200,
    facecolor="white",
)
