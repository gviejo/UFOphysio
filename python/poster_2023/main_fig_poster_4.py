# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-15 17:49:22

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


colors = {"ADN": "#EA9E8D", "LMN": "#8BA6A9", "PSB": "#CACC90", "ctrl":"grey"}



###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, figure=fig, hspace=0.5, height_ratios=[0.5, 0.25])


gs_psb = gridspec.GridSpecFromSubplotSpec(4, 2, 
    subplot_spec=outergs[0,0], width_ratios=[0.3, 0.7], height_ratios=[0.2, 0.2, 0.2, 0.2], hspace=0.3
    )

sessions = ['LMN-ADN/A5044/A5044-240402A', 'LMN-PSB/A3018/A3018-220613A']

exs = [5211.327, 325.537]
lws = 0.5
chs = [6, 1]
structs = ['adn', 'psb']
bounds = [0.02, 0.8]

for i, struct in enumerate(['adn', 'psb']):
    path = os.path.join(data_directory, sessions[i])
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(data.path)
    fp, timestep = get_memory_map(os.path.join(data.path, data.basename+".dat"), data.nChannels, 20000)
    lfp = nap.TsdFrame(t=timestep, d=fp)
    channels = data.group_to_channel


    tcurves = { struct:nap.compute_1d_tuning_curves(data.spikes.getby_category("location")[struct], data.position['ry'], 60, data.epochs['wake']),
                "lmn":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['lmn'], data.position['ry'], 60, data.epochs['wake'])}
    for k in tcurves:
        tcurves[k] = smoothAngularTuningCurves(tcurves[k], 40, 5)
        # mi = nap.compute_1d_mutual_info(tcurves[k], data.position['ry'], data.epochs['wake'])
        # tcurves[k] = tcurves[k][mi[mi>0.1].dropna().index]

    ep = nap.IntervalSet(exs[i] - bounds[i], exs[i] + bounds[i])
    
    tmp = lfp.restrict(ep)
    tmp = nap.TsdFrame(t=tmp.t, d=tmp.d.astype(float))

    # LMN LFP
    subplot(gs_psb[0,i])
    noaxis(gca())

    [plot((tmp[:,c]-k*1000)*100, linewidth=lws, color=colors["LMN"]) for k, c in enumerate(channels[chs[i]])]

    xlim(ep[0,0], ep[0,1])

    if i == 0: ylabel("LMN", rotation=0, labelpad=25)
    
    ms = 2
    mew = 1

    # LMN Spikes
    subplot(gs_psb[1,i])
    simpleaxis(gca())
    gca().spines['bottom'].set_visible(False)    
    grs = data.spikes.getby_category("location")['lmn'].restrict(ep)
    grs['order'] = np.argsort(tcurves['lmn'].idxmax()).values
    plot(grs.to_tsd("order"), '|', color=colors['LMN'], markersize=ms, markeredgewidth=mew)
    xlim(ep[0,0], ep[0,1])
    xticks([])
    yticks([len(grs)-1], [len(grs)])

    # SPIKEs
    if i == 0:
        subplot(gs_psb[2,i])
    else:
        subplot(gs_psb[2:,i])
    simpleaxis(gca())
    gca().spines['bottom'].set_visible(False)        
    grs = data.spikes.getby_category("location")[structs[i]].restrict(ep)
    grs['order'] = np.argsort(tcurves[struct].idxmax()).values
    plot(grs.to_tsd("order"), '|', color=colors[structs[i].upper()], markersize=ms, markeredgewidth=mew)
    xlim(ep[0,0], ep[0,1])
    ylabel(structs[i].upper(), rotation=0, labelpad=12, y=0.4)
    xticks([])
    yticks([len(grs)-1], [len(grs)])
    
    gca().spines['bottom'].set_visible("True")
    if i == 0:
        gca().spines['bottom'].set_bounds(ep.end[0]-0.01, ep.end[0])
        xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 ms"])
    else:
        gca().spines['bottom'].set_bounds(ep.end[0]-0.2, ep.end[0])
        xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["200 ms"])
        

#####################################
# CROSS_CORR
datatosave = cPickle.load(open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/CC_UFO_PSB.pickle"), 'rb'))
ccs_long = datatosave['ccs_long']
ccs_short = datatosave['ccs_short']

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, 
    subplot_spec=outergs[1,0],
    wspace = 0.4
    )


gs2_1 = gridspec.GridSpecFromSubplotSpec(1, 2, 
    subplot_spec=gs2[0,0], hspace = 1.2, wspace=0.4
    )
gs2_2 = gridspec.GridSpecFromSubplotSpec(1, 2, 
    subplot_spec=gs2[0,1], hspace = 1.2, wspace=0.4
    )

xpos = [0, 1]
ypos = [0, 0]
# titles = ["Wake", "REM sleep", "nREM sleep"]
titles = ["Wake", "nREM sleep"]

for i, e in enumerate(['wak', 'sws']):
    subplot(gs2_1[0, i])
    simpleaxis(gca())
    for j, s in enumerate(['lmn', 'adn']):
        plot(ccs_short[s][e].mean(1), color = colors[s.upper()], linewidth=1)
    xlabel("Wave time (s)")
    if i in [0, 2]:
        ylabel("Rate\n(norm.)", rotation = 0, labelpad=15, y=0.3)
    title(titles[i])

    subplot(gs2_2[0, i])
    simpleaxis(gca())
    plot(ccs_long['psb'][e].mean(1), color = colors["PSB"], linewidth=1)
    xlabel("Wave time (s)")
    if i in [0, 2]:
        ylabel("Rate\n(norm.)", rotation = 0, labelpad=15, y=0.3)

    title(titles[i])

# axes = {}



# for i, s in enumerate(['lmn', 'adn', 'psb']):
#     axes[s] = {}
#     for j, e in enumerate(ccs_long[s].keys()):        
#         subplot(gs2[i,j])
#         simpleaxis(gca())
#         axes[s][e] = gca()
#         idx = ccs_long[s][e].index.values
#         m = ccs_long[s][e].mean(1).values
#         d = ccs_long[s][e].std(1).values
#         cc = ccs_long[s][e].loc[-0.5:0.5]
#         plot(cc.mean(1), color = colors[s.upper()], linewidth=lws)
#         # plot(cc, color = colors[s.upper()], linewidth=0.1)
#         # fill_between(idx, m-d, m+d, alpha=0.2, color=colors[s.upper()])

#         if s != 'psb':
#             xticks([])

#         if j == 0:
#             ylabel(s.upper(), rotation=0, labelpad=15, y=0.4)

# for s in axes.keys():
#     maxv=np.max([axes[s][e].get_ylim()[1] for e in axes[s].keys()])
#     minv=np.min([axes[s][e].get_ylim()[0] for e in axes[s].keys()])
#     for e in axes[s].keys():
#         axes[s][e].set_ylim(minv, maxv)

outergs.update(top=0.98, bottom=0.08, right=0.98, left=0.1)

savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/FENS 2024/figures/fig4.pdf",
    dpi=200,
    facecolor="white",
)
