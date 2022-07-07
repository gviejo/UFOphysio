# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-10 14:21:59
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-13 14:51:54
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *

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

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

ac_wake = []
ac_sws = []

#session = 'LMN-ADN/A5002/A5002-200303B'
session = 'LMN-ADN/A5011/A5011-201014A'

############################################################################################### 
# LOADING DATA
###############################################################################################
path = os.path.join(data_directory, session)
data = nap.load_session(path, 'neurosuite')
spikes = data.spikes
position = data.position
wake_ep = data.epochs['wake']
sws_ep = data.read_neuroscope_intervals('sws')
#rem_ep = data.read_neuroscope_intervals('rem')
ufo_ep, ufo_ts = loadUFOs(path)

idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
spikes = spikes[idx]

meanwaveforms = cPickle.load(open(os.path.join(path, 'pynapplenwb', 'MeanWaveForms.pickle'), 'rb'))
waveforms = {}
maxch = {}
for c in meanwaveforms.keys():
    for n in meanwaveforms[c].columns:
        if n in list(spikes.keys()):
            wave = meanwaveforms[c][n].values
            nch = len(wave)//32
            wave = wave.reshape(32, nch)
            waveforms[n] = wave
            maxch[n] = np.argmax(np.abs(wave).max(0))



############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

###############################################################################################
# MEMORY MAP
###############################################################################################
data.load_neurosuite_xml(data.path)
channels = data.group_to_channel[np.unique(spikes._metadata["group"].values)[0]]

filename = data.basename + ".dat"
        
fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)

dummy = pd.Series(index=timestep, data= np.arange(len(timestep)))

nspikes = {}
for i, t in enumerate(ufo_ts.index.values[0:200]):
    idx = dummy.loc[t-0.005:t+0.005]
    ep = nap.IntervalSet(start=idx.index[0],end=idx.index[-1])
    spk = spikes.restrict(ep)
    n_spk = np.sum([len(spk[n]) for n in spk.keys()])
    nspikes[t] = n_spk
nspikes = pd.Series(nspikes)

nspikes = nspikes.sort_values(ascending=False)


#for i, t in enumerate(ufo_ts.index.values[0:200]):
for i, t in enumerate(nspikes.index.values):
    idx = dummy.loc[t-0.005:t+0.005]
    ep = nap.IntervalSet(start=idx.index[0],end=idx.index[-1])
    spk = spikes.restrict(ep)

    print(i)
    fig = figure(figsize=(22,18))
    gs = GridSpec(2,2, width_ratios=[0.4, 0.6])

    # waveforms
    gs3 = gridspec.GridSpecFromSubplotSpec(1, len(spikes), gs[0,0])
    for k, n in enumerate(tuning_curves.idxmax().sort_values().index.values[::-1]):
        subplot(gs3[0,k])
        noaxis(gca())
        wave = waveforms[n]
        for j in range(nch):
            plot(np.arange(0, len(wave)), 15*wave[:,j]+j*5000, linewidth = 1,
                color = hsv_to_rgb([tuning_curves[n].idxmax()/(2*np.pi),1,1]))
        ylim(-5000, 5000*(nch+2))

    # tuning curves
    gs2 = gridspec.GridSpecFromSubplotSpec(1, len(spikes), gs[1,0])
    for j, n in enumerate(tuning_curves.idxmax().sort_values().index.values[::-1]):
        subplot(gs2[0,j])                
        noaxis(gca())
        clr = hsv_to_rgb([tuning_curves[n].idxmax()/(2*np.pi),1,1])        
        fill_betweenx(tuning_curves[n].index.values,
            np.zeros_like(tuning_curves[n].index.values),
            tuning_curves[n].values,
            color = clr
            )
        xticks([])
        yticks([])
        ylim(0, 2*np.pi)


    # lfp
    ax2 = subplot(gs[0,1])
    noaxis(gca())    
    lfp = nap.TsdFrame(t=idx.index.values,d=fp[idx.values, :][:, channels],columns=channels)
    axvline(t)
    for j, c in enumerate(lfp.columns):
        plot(lfp[c].as_series()+j*600, color = 'grey')

    
    # rasters        
    subplot(gs[1,1], sharex = ax2)
    noaxis(gca())    
    axvline(t)
    for j, n in enumerate(spikes.keys()):
        spk = spikes[n].restrict(ep).index.values
        if len(spk):            
            clr = hsv_to_rgb([tuning_curves[n].idxmax()/(2*np.pi),1,1])
            plot(spk, np.ones_like(spk)*tuning_curves[n].idxmax(), '|', color = clr, markersize = 10, markeredgewidth = 3, alpha = 1)

            ####
            #ax2.plot()
            for s in spk:
                ep_spk = nap.IntervalSet(start=s-0.0002, end=s+0.0002)
                lfpspk = lfp[lfp.columns[maxch[n]]].restrict(ep_spk)
                ax2.plot(lfpspk.as_series()+maxch[n]*600, color = clr)


    ylim(0, 2*np.pi)
    #tight_layout()
 

    savefig(os.path.join('../figures/', data.basename+'_'+str(i)+'.pdf'))
    close(fig)
    

os.system("/snap/bin/pdftk ../figures/"+data.basename+"_*.pdf cat output ../figures/UFOs_"+data.basename+"_decoding.pdf")
os.system("rm ../figures/"+data.basename+"_*.pdf")




