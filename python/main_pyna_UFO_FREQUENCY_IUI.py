# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-14 16:14:23

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
# from functions import *
# import pynacollada as pyna
from ufo_detection import *
from matplotlib.pyplot import *

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

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


rates = {}
iui = {'wak':{}, 'rem':{}, 'sws':{}}

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)    
    
    r = pd.Series(index=['wak', 'rem', 'sws'], data = np.NaN)    

    if ufo_ts is not None:      
        for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
            r[e] = ufo_ts.restrict(ep).rate
            tmp = np.hstack([np.diff(ufo_ts.restrict(ep.loc[[i]]).t) for i in ep.index])
            count, bins = np.histogram(tmp, np.geomspace(0.001, 100, 50))
            count = count.astype('float')/float(np.sum(count))
            iui[e][s] = pd.Series(index=bins[0:-1], data=count)

    rates[s.split("/")[-1]] = r

rates = pd.DataFrame.from_dict(rates).T

for e in iui.keys():
    iui[e] = pd.DataFrame.from_dict(iui[e])

datatosave = {"rates":rates, "iui":iui}

import _pickle as cPickle
cPickle.dump(datatosave, open("/mnt/home/gviejo/Dropbox/UFOPhysio/figures/poster/fig1.pickle", 'wb'))

figure(figsize = (8, 6))
rcParams.update({'font.size': 20})

for i, e in enumerate(rates.columns):
    y = rates[e].dropna().values
    x = np.ones(len(y))*i+np.random.randn(len(y))*0.1
    plot(x, y, 'o')
    xticks(range(3), rates.columns)

    plot([i], [y.mean()], 'o', color = 'red', markersize=10)


savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/Frquency_UFO.png"))

figure(figsize = (20, 5))

for i, e in enumerate(iui.keys()):

    subplot(1,3,i+1)
    semilogx(iui[e], color='grey', alpha=0.5, linewidth=0.5)
    semilogx(iui[e].mean(1), color='red', linewidth=3)
    title(e)
    xticks([0.01, 0.1, 1, 100], [0.01, 0.1, 1, 100])
    xlabel("Inter UFO (s)")

tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/IUI_UFO.png"))

show()

