# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-05-01 14:35:04
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-18 16:32:29

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
    fig_height = fig_width*golden_mean*0.95         # height in inches
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
# sws_ex = nap.IntervalSet(3095.8, 3097.3)
sws_ex = nap.IntervalSet(11005.0,11005.0+1.5)
rem_ex = nap.IntervalSet(5198.114-0.2, 5198.114+0.2)

ts_ex = [11005.398, 11005.982]

ts_wak = [7326.612]
ts_rem = [5198.114]

tcurves = { "ADN":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['adn'], data.position['ry'], 60, data.epochs['wake']),
            "LMN":nap.compute_1d_tuning_curves(data.spikes.getby_category("location")['lmn'], data.position['ry'], 60, data.epochs['wake'])}


for k in tcurves:
    tcurves[k] = smoothAngularTuningCurves(tcurves[k], 40, 5)
    mi = nap.compute_1d_mutual_info(tcurves[k], data.position['ry'], data.epochs['wake'])
    tcurves[k] = tcurves[k][mi[mi>0.1].dropna().index]


# plt.figure()
# for i in range(15):
#     plt.subplot(3,5,i+1)
#     plt.plot(tcurves['ADN'].iloc[:,i])
#     plt.title(mi.values.flatten()[i])
# plt.show()



structs = ['Thalamus', 'Mamillary Body']

###############################################################################################################
# PLOT
###############################################################################################################

fig = figure(figsize=figsize(0.9))

outergs = GridSpec(2, 1, figure=fig, height_ratios=[0.6, 0.2], hspace=0.2)

gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.4, 0.03, 0.5])

#####################################
# HISTO 
gs_hs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0,0], hspace=0.05)

files = [
    os.path.join(os.path.expanduser("~"),"Dropbox/UFOPhysio/figures/poster/A5044/A5044_S2_8_4xDAPI.png"),
    os.path.join(os.path.expanduser("~"),"Dropbox/UFOPhysio/figures/poster/A5044/A5044_S8_6_4xDAPI.png")]

files = [
    os.path.join(os.path.expanduser("~"),"Dropbox/UFOPhysio/figures/poster/ADN_histo_graph.png"),
    os.path.join(os.path.expanduser("~"),"Dropbox/UFOPhysio/figures/poster/LMN_histo_graph.png")]


for i, f in enumerate(files):
    subplot(gs_hs[i, 0])
    noaxis(gca())
    img = mpimg.imread(f)
    imshow(img, aspect="equal")
    # title(structs[i])
    xticks([])
    yticks([])




#####################################
# LFP SWS Examples

gs_lfp_2 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0,2], height_ratios=[0.2, 0.2, 0.1, 0.2], width_ratios=[0.9, 0.02], wspace=0.1)

chs = [0, 6]
yls = ['ADN', 'LMN']
structs = ['ADN', 'ctrl', 'LMN', 'ctrl']
lws = [0.5, 0.5, 0.5, 0.5]
alphas = [1, 0.5, 1, 0.5]


# for i, t in enumerate(ts_ex):
t = ts_ex[1]
tmp = lfp.get(t-0.02, t+0.04)

# gs_ex = gridspec.GridSpecFromSubplotSpec(4, len(ts_ex), subplot_spec=gs1[i])#, hspace=0.0, wspace=0.0)
for j in range(2):
    subplot(gs_lfp_2[j,0])
    noaxis(gca())
    [plot((tmp[:,c]-k*1000)*2, linewidth=lws[j], color=colors[structs[j]]) for k, c in enumerate(channels[chs[j]])]
    xlim(t-0.02, t+0.04)
    ylabel(yls[j], rotation=0, y=0.3, labelpad=15)

A = nap.load_file(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/"+name.split("/")[-1]+"_TFD_Ex.npz"))
freq = A.columns.astype("int")
idx = (freq>=100) & (freq <= 3000)
freq2 = freq[idx]
tmp = A[:,idx].get(t-0.02, t+0.04).values
tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)


# Power
subplot(gs_lfp_2[2,0])
simpleaxis(gca())
plot(nSS.get(t-0.02, t+0.04), color = colors["LMN"], linewidth=0.8, label="600-2000 Hz")
xlim(t-0.02, t+0.04)
gca().spines['bottom'].set_bounds(t+0.04, t+0.04)
# xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 ms"])
xticks([])
ylabel("Power\n(z)", rotation=0, labelpad=25, y=0.4)
axhline(5.0, linewidth=0.5, color=COLOR, linestyle="--")
# legend(frameon=False, bbox_to_anchor=(0, -1), handlelength=0.0, loc=3)

# Time Frequency decomposition
subplot(gs_lfp_2[3,0])
simpleaxis(gca())
img = imshow(tmp.T, 
    aspect='auto', origin='lower', extent=(t-0.02, t+0.04,float(freq2[0]),float(freq2[-1])),
    interpolation='bilinear', cmap = "afmhot"
    )
# axhline(500.0, linewidth=0.05, color='white')
# axhline(1000.0, linewidth=0.1, color='white')
gca().spines['bottom'].set_bounds(t+0.03, t+0.04)
xticks(gca().spines['bottom'].get_bounds()[0] + np.diff(gca().spines['bottom'].get_bounds())/2, ["10 ms"])
xlim(t-0.02, t+0.04)
ylabel("Freq.\n(Hz)", rotation=0, labelpad=15, y=0.4)
# axhline(3.0, linewidth=0.5, color=COLOR, linestyle="--")
# legend(frameon=False, bbox_to_anchor=(0, -1), handlelength=0.0, loc=3)

ax = subplot(gs_lfp_2[3,1])
colorbar(img, ax)
title("z")

###########################################
gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outergs[1, 0], width_ratios=[0.4, 0.01, 0.2, 0.4], hspace=0.1)

#####################################
# TUNING CURVES

gs_tc = gridspec.GridSpecFromSubplotSpec(
    4, 3, subplot_spec=gs1[0,0],
    width_ratios = [0.1, 0.3, 0.2], 
    height_ratios = [0.0, 0.2, 0.2, 0.0], 
    hspace=0.7,
    wspace=0.4
)

for i, st in enumerate(["ADN", "LMN"]):

    tcurve = tcurves[st]
    peak = pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]))
    order = peak.sort_values().index.values
        

    # Tunning curves centerered
    subplot(gs_tc[i+1, 1])
    simpleaxis(gca())    
    tc = centerTuningCurves(tcurves[st])
    tc = tc / tc.loc[0]
    plot(tc, linewidth=0.1, color=colors[st], alpha=0.4)
    plot(tc.mean(1), linewidth=0.5, color=colors[st])
    xticks([])
    if i == 1:
        xticks([-np.pi, 0, np.pi], [-180, 0, 180])
        xlabel("Centered HD")
    ylabel(st, rotation=0, labelpad=15)

    # All directions as arrows
    subplot(gs_tc[i+1,2], aspect="equal")
    gca().spines["left"].set_position(("data", 0))
    gca().spines["bottom"].set_position(("data", 0))
    gca().spines["top"].set_visible(False)
    gca().spines["right"].set_visible(False)

    peaks = tcurves[st].idxmax()
    theta = peaks.values
    radius = np.sqrt(tcurves[st].max(0).values)
    for t, r in zip(theta, radius):
        arrow(
            0, 0, np.cos(t), np.sin(t), linewidth=0.1, color=colors[st], width=0.1
        )

    xticks([-2, 2], ["90°", "0°"])
    xlim(-2, 2)
    ylim(-2, 2)
    yticks([])





######################################
# RATES
data2 = cPickle.load(open("/mnt/home/gviejo/Dropbox/UFOPhysio/figures/poster/fig1.pickle", 'rb'))

# gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[0,1], hspace = 0.2, wspace = 0.5)

# gs_rates = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3[0,0], hspace = 0.5, wspace = 0.5)#, height_ratios=[0.5, 0.2], width_ratios=[0.1, 0.4])


subplot(gs1[0,2])
simpleaxis(gca())

titles = ['NREM', 'WAKE']
colors2 = ['#686963', '#8aa29e', '#3d5467']

rates = data2['rates']
for i, e in enumerate(['sws', 'wak']):
    y = rates[e].dropna().values
    x = np.ones(len(y))*i+np.random.randn(len(y))*0.1
    plot(y, x, 'o', markersize=0.5, color = colors2[i])
    plot([y.mean(), y.mean()], [i-0.2, i+0.2], '-', color = 'red', linewidth=1.0)
    gca().invert_yaxis()
    xlim(0, 2)
yticks(range(2), titles)
# xticks([0, 1, 2, 3], ["0", "1", "2", "3"])
xlabel("Rate (Hz)")

gs_iui = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs1[0,3], hspace = 0.5, wspace = 0.5, width_ratios=[0.1, 0.6, 0.1])


iui = data2['iui']
for i, e in enumerate(['sws', 'wak']):
    subplot(gs_iui[i,1])
    simpleaxis(gca())        
    # semilogx(iui[e], color='grey', alpha=0.5, linewidth=0.5)
    tmp = iui[e].mean(1)*100
    bar(tmp.index.values[0:-1], tmp.values[0:-1], width=np.diff(tmp.index.values), align='edge', facecolor = colors2[i], edgecolor=colors2[i])
    gca().set_xscale("log")
    # semilogx(, linewidth=1, color=colors2[i])
    # gca().text(1.1, 0.3, titles[i], transform=gca().transAxes)
    yticks([5])
    ylabel("%", rotation=0, labelpad=10)
    if i == 0:         
    # if i in [0, 1]:
        gca().spines['bottom'].set_visible(False)
        xticks([])
    else:
        xticks([0.01, 1, 100], [0.01, 1, 100])
        xlabel("Inter waves intervals (s)")





outergs.update(top=0.98, bottom=0.09, right=0.98, left=0.05)


savefig(
    os.path.expanduser("~") + r"/Dropbox/Applications/Overleaf/FENS 2024/figures/fig1.pdf",
    # os.path.expanduser("~") + "/Dropbox/UFOPhysio/figures/poster/fig1.pdf",
    dpi=200,
    facecolor="white",
)
# show()



# plt.figure()
# ax = plt.subplot(211)
# plt.subplot(211)
# plt.plot(np.random.randn(1000))
# plt.subplot(212, sharex = ax)
# plt.plot(np.random.randn(1000)*1000)
# plt.show()