# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-18 17:59:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-12-11 10:58:31
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *

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

datasets = {
    'adn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    'psb':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')
}

ccs = {r:{e:[] for e in ['up', 'down']} for r in ['adn', 'psb']}

for r in datasets.keys():
    for s in datasets[r]:

        print(s)
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, s)
        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        
        ufo_ep, ufo_ts = loadUFOs(path)        

        try:
            up_ep = data.read_neuroscope_intervals('up')
            up_ts = up_ep.starts
            down_ep = data.read_neuroscope_intervals('down')
            down_ts = down_ep.get_intervals_center()
        except:
            print("No up in "+s)
            up_ts = None

        if ufo_ts is not None and up_ts is not None:
            for e, ts in zip(['up', 'down'], [up_ts, down_ts]):
                grp = nap.TsGroup({0:ts,1:ufo_ts}, evt = np.array([e, 'ufo']))

                cc = nap.compute_crosscorrelogram(grp, 0.01, 1, sws_ep)
                cc.columns = [s.split("/")[-1]]

                ccs[r][e].append(cc)


for r in ccs.keys():
    for e in ccs[r].keys():
        ccs[r][e] = pd.concat(ccs[r][e], 1)


rcParams.update({'font.size': 15})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
lw = 2

figure(figsize = (8, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

gs = GridSpec(3,2)
for i, r in enumerate(['adn', 'psb']):
    for j, e in enumerate(ccs[r].keys()):
        subplot(gs[i,j])
        if i == 0:
            title(e)
        if j == 0:
            ylabel(r, rotation=0, labelpad = 30)
        tmp = ccs[r][e].values
        tmp = tmp - tmp.mean(0)
        tmp = tmp / tmp.std(0)
        imshow(tmp.T, aspect='auto', cmap = 'jet')
        x = ccs[r][e].index.values
        xticks([0, len(x)//2, len(x)], [x[0], 0.0, x[-1]])

for j, e in enumerate(ccs[r].keys()):
    subplot(gs[-1,j])
    for i, r in enumerate(['adn', 'psb']):
        plot(ccs[r][e].mean(1), color = colors[i+1], linewidth=lw, label = r)
    axvline(0.0)
    xlim(x[0], x[-1])
    legend(frameon=False)
    xlabel("CC "+e+"/ufo")
tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/ALL_CC_UFO_UPDOWN.png"))


figure(figsize = (6, 12))
tmp = ccs['adn']['up']
animals = np.array([i.split("-")[0] for i in tmp.columns])

for i, a in enumerate(np.unique(animals)):
    subplot(len(np.unique(animals)), 1, i+1)
    plot(tmp[tmp.columns[animals==a]])   
    xlabel("CC "+e+"/ufo")
tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/CC_UFO_UPDOWN_ADN_SESSIONS.png"))
