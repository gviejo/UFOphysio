# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-06-09 18:46:16

import numpy as np
import pandas as pd
import pynapple as nap
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
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_MMN.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

alldata = {
    'mmn_cc' : [],
    'lmn_cc' : [],
    'mua_cc' : [],
    'lmn_au' : [],
    'mmn_au' : [],
    'ufo_mua_lmn_cc' : [],
    'ufo_mua_mmn_cc' : []    
}


for s in datasets:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)

    if ufo_ep is not None:
        
        group_lmn = spikes._metadata[spikes._metadata["location"].str.contains("lmn")]["group"].max()
        idx_lmn = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
        spikes_lmn = spikes[idx_lmn]

        spikes_mmn = spikes.getby_threshold("group", 4, op=">=")

        # MUA        
        mua_mmn = np.sort(np.hstack([spikes_mmn[n].index.values for n in spikes_mmn.keys()]))
        mua_lmn = np.sort(np.hstack([spikes_lmn[n].index.values for n in spikes_lmn.keys()]))

        mua = nap.TsGroup({
            0:nap.Ts(mua_lmn),
            1:nap.Ts(mua_mmn)
            })


        ############################################################################################### 
        ###############################################################################################
        bin_size = 0.5
        window_size = 60
        sws_ep = data.read_neuroscope_intervals('sws')
        alldata['mmn_cc'].append(nap.compute_eventcorrelogram(spikes_mmn, ufo_ts, bin_size, window_size, sws_ep))
        alldata['lmn_cc'].append(nap.compute_eventcorrelogram(spikes_lmn, ufo_ts, bin_size, window_size, sws_ep))

        alldata['mua_cc'].append(nap.compute_crosscorrelogram(mua, bin_size, window_size, sws_ep))

        autocorr = nap.compute_autocorrelogram(mua, bin_size, window_size, sws_ep)
        alldata['lmn_au'].append(autocorr[0])
        alldata['mmn_au'].append(autocorr[1])
        
        ufo_cc = nap.compute_eventcorrelogram(mua, ufo_ts, bin_size, window_size, sws_ep)
        alldata['ufo_mua_lmn_cc'].append(ufo_cc[0])
        alldata['ufo_mua_mmn_cc'].append(ufo_cc[1])


for k in alldata.keys():
    alldata[k] = pd.concat(alldata[k], 1)
    alldata[k] = alldata[k].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
        



figure()
xlabels = ['ufo/lmn', 'ufo/mmn', 'ufo/MUA(lmn)', 'ufo/MUA(mmn)']
for i,k in enumerate(['lmn_cc', 'mmn_cc', 'ufo_mua_lmn_cc', 'ufo_mua_mmn_cc']):
    subplot(2,2,i+1)
    cc = alldata[k] 
    plot(cc, alpha =0.5, color = 'grey')
    plot(cc.mean(1), linewidth=1.5)
    xlabel(xlabels[i])


figure()
xlabels = ['MUA(LMN) / MUA(MMN)', 'MUA(LMN) / MUA(LMN)', 'MUA(MMN) / MUA(MMN)']
for i,k in enumerate(['mua_cc', 'lmn_au', 'mmn_au']):
    subplot(2,2,i+1)    
    cc = alldata[k]
    plot(cc, alpha =0.5, color = 'grey')
    plot(cc.mean(1), linewidth=1.5)
    xlabel(xlabels[i])

show()