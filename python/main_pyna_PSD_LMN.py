# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-02-15 14:29:17

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from ufo_detection import *
from scipy import signal
from multiprocessing import Pool
import functools
from mne.time_frequency import psd_array_multitaper

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

datasets = {'lmn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
            'adn':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),            
            'psb':np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),            
            }

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}

psd = {'wak':{}, 'rem':{}, 'sws':{}}

# for g in datasets.keys():
for g in ["lmn"]:
# for g in ['ca1']:
    
    for s in datasets[g]:

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
        rem_ep = data.read_neuroscope_intervals('rem')
        ufo_ep, ufo_ts = loadUFOs(path)

        if ufo_ts is not None:        

            ###############################################################################################
            # MEMORY MAP
            ###############################################################################################
            data.load_neurosuite_xml(data.path)
            channels = data.group_to_channel

            if g == "lmn":
                sign_channels = channels[ufo_channels[s][0]]
            else:
                sign_channels = channels[np.unique(spikes.getby_category("location")[g].get_info("group"))[0]]

            lfp = data.load_lfp(channel=sign_channels, extension='.eeg', frequency=1250.0)

            

            for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
                tmp = []
                for c in sign_channels:
                    print(s, e, c)
                    # pxx, f = psd_array_multitaper(lfp.loc[c].restrict(ep).values, 1250.0, adaptive=True,
                    #                         normalization='full', verbose=0)
                    f, pxx = signal.welch(lfp.loc[c].restrict(ep).values, 1250.0)
                    tmp.append(pxx)
                tmp = np.array(tmp)
                psd[e][s] = pd.Series(index=f, data=np.mean(tmp, 0))


for e in psd.keys():
    psd[e] = pd.DataFrame.from_dict(psd[e])


figure()
for i,e  in enumerate(psd.keys()):
    subplot(1,3,i+1)
    semilogx(psd[e].loc[1:100].mean(1))
    title(e)
show()