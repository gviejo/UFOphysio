# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-10 12:39:17

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
sys.path.append("../")
from functions import *
# import pynacollada as pyna
from ufo_detection import *
import yaml

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

alldatasets = yaml.safe_load(open(os.path.join(data_directory,'datasets_SOUND.yaml'), 'r'))

datasets = alldatasets["click_10s"] + alldatasets["click_5s"]

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


ccs = {'wak':[], 'sws':[]}

peths = {}

for s in datasets:
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
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]

    
    ufo_ep, ufo_tsd = loadUFOs(path)

    analogin, ts = get_memory_map(os.path.join(data.path, data.basename+"_0_analogin.dat"), 2, frequency=20000)
    peaks,_ = scipy.signal.find_peaks(np.diff(analogin[:,0]), height=2000)
    peaks+=1    

    ttl = nap.Ts(t=ts[peaks])
    ttl.save(data.path + "/ttl_sound")

    for e, ep in zip(['wak', 'sws'], [wake_ep, sws_ep]):
        cc = nap.compute_eventcorrelogram(
            nap.TsGroup({0:ufo_tsd}), 
            ttl, 0.001, 0.1, ep = ep, norm=True)

        ccs[e].append(cc)#/len(ttl.restrict(ep))

    peth = nap.compute_perievent(
        nap.TsGroup({0:ufo_tsd.restrict(sws_ep)}), 
        ttl.restrict(sws_ep), 0.1)

    peths[s] = peth[0]

    # Writing for neuroscope
    peaks2 = ttl.as_units('ms').index.values
    datatowrite = peaks2
    n = len(peaks2)
    texttowrite = np.repeat(np.array(['SOUND 1']), n)    
    evt_file = os.path.join(data.path, data.basename+'.evt.py.snd')
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()

for e in ccs.keys():
    ccs[e] = pd.concat(ccs[e], 1)

figure()
subplot(211)
for s in peths.keys():
    plot(peths[s].to_tsd(), 'o')
ylabel("Trial")
axvline(0)
xlim(-0.1, 0.1)
subplot(212)
plot(ccs['sws'], label = "UFO")
xlim(-0.1, 0.1)
axvline(0)
xlabel("Sound time")
ylabel("UFO/SOUND")
tight_layout()
show()

# Saving
datatosave = {
    "ccs":ccs,
    "peths":{s:peths[s].to_tsd().as_series() for s in peths.keys()}
}

import _pickle as cPickle
cPickle.dump(datatosave, open(os.path.expanduser("~/Dropbox/UFOPhysio/figures/poster/cc_sound.pickle"), 'wb'))
