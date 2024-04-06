# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-03-21 13:21:51
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-03-21 16:02:59

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations

# import pynacollada as pyna
import sys
sys.path.append("../")
from ufo_detection import *
from functions import *


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


path = "/mnt/ceph/users/gviejo/OPTO/B3700/B3702/B3702-240313A"



fp, timestep = get_memory_map(os.path.join(path, "B3702-240313A.dat"), 40)


sign_channels = np.array([29, 19, 27, 23, 26, 28, 25, 24])
ctrl_channels = np.array([1, 15, 0, 14, 9, 13, 10, 11])

ufo_ep, ufo_tsd = detect_ufos_v2(fp, sign_channels, ctrl_channels, timestep)


n_channels = 2
file = os.path.join(path, "B3702-240313A_0_analogin.dat")
f = open(file, 'rb')
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
bytes_size = 2        
n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
f.close()
with open(file, 'rb') as f:
	data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
data = data[:,0].astype(np.int32)

fs = 20000

peaks,_ = scipy.signal.find_peaks(np.diff(data), height=1000)
peaks+=1
timestep = np.arange(0, len(data))/fs
# # analogin = pd.Series(index = timestep, data = data)

ttl = nap.Ts(t=timestep[peaks])

ep = pd.read_csv(os.path.join(path, "Epoch_TS.csv"), header=None)

sleep_ep = nap.IntervalSet(start=ep.loc[0,0], end=ep.loc[0,1])

cc = nap.compute_eventcorrelogram(
	nap.TsGroup({0:ufo_tsd}), 
	ttl, 0.005, 0.1, ep = sleep_ep)

peth = nap.compute_perievent(
	nap.TsGroup({0:ufo_tsd.restrict(sleep_ep)}), 
	ttl.restrict(sleep_ep), 0.1)

figure()
subplot(211)
plot(peth[0].to_tsd(), 'o')
ylabel("Trial")
axvline(0)
xlim(-0.1, 0.1)
subplot(212)
plot(cc, label = "UFO")
xlim(-0.1, 0.1)
axvline(0)
xlabel("Sound time")
ylabel("UFO/SOUND")
tight_layout()

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_SOUND_B3702.png"))

###########################################################################################################
# Writing for neuroscope
start = ufo_ep.as_units('ms')['start'].values
peaks = ufo_tsd.as_units('ms').index.values
ends = ufo_ep.as_units('ms')['end'].values

datatowrite = np.vstack((start,peaks,ends)).T.flatten()

n = len(ufo_ep)

texttowrite = np.vstack(((np.repeat(np.array(['UFO start 1']), n)),
                        (np.repeat(np.array(['UFO peak 1']), n)),
                        (np.repeat(np.array(['UFO stop 1']), n))
                            )).T.flatten()
basename = "B3702-240313A"
evt_file = os.path.join(path, basename + '.evt.py.ufo')
f = open(evt_file, 'w')
for t, n in zip(datatowrite, texttowrite):
    f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
f.close()


peaks2 = ttl.as_units('ms').index.values
datatowrite = peaks2

n = len(ttl)

texttowrite = np.repeat(np.array(['SOUND 1']), n)

basename = "B3702-240313A"
evt_file = os.path.join(path, basename + '.evt.py.snd')
f = open(evt_file, 'w')
for t, n in zip(datatowrite, texttowrite):
    f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
f.close()  