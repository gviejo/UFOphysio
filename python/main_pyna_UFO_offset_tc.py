import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import sys, os

from matplotlib import pyplot as plt
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations

from sklearn.decomposition import KernelPCA
from matplotlib.colors import hsv_to_rgb

import sys
sys.path.append("..")

from ufo_detection import get_memory_map, loadUFOs
from functions.functions import *

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

mdpec = {}

ufo_channels = np.genfromtxt(os.path.join(data_directory, 'channels_UFO.txt'), delimiter = ' ', dtype = str, comments = '#')
ufo_channels = {a[0]:a[1:].astype('int') for a in ufo_channels}


offset_si = {}
offset_r = {}

for s in datasets:
# for s in ["LMN-ADN/A5043/A5043-230301A"]:
# for s in ["LMN-ADN/A5043/A5043-230306A"]:

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

    nSS = nap.load_file(os.path.join(data.path, "nSS_LMN.npz"))
    ufo_ts = ufo_ts.value_from(nSS)
    ufo_ep = ufo_ep[ufo_ts > 8]
    ufo_ts = ufo_ts[ufo_ts > 8]

    spikes = spikes[spikes.location == "adn"]
    # spikes = spikes[(spikes.location=="adn") | (spikes.location=="lmn")]

    if len(ufo_ts):


        # UFOs during wake
        ufo_wake = ufo_ts.restrict(wake_ep)
        
        offset_si_all = []
        ry = position['ry']
        offset_values = np.linspace(-2, 2, 200)
        offset_si_all = []
        offset_tc_all = []
        offset_r_all = []

        for offset in offset_values:
            new_ry = nap.Tsd(t=ry.t-offset, d=ry.values, 
                time_support=ry.time_support)
            new_ry = smoothAngle(new_ry, 2)
            # plot(new_ry, color=cm.viridis((offset - offset_values.min()) / (offset_values.max() - offset_values.min())))

            # control 
            tc = nap.compute_tuning_curves(
            	nap.TsGroup({0: ufo_wake}), new_ry, 30, range=(0, 2*np.pi), epochs=new_ry.time_support, feature_names=["angle"]
            	)
            tc = smoothAngularTuningCurves(tc, 20, 1)
            offset_si_all.append(nap.compute_mutual_information(tc)['bits/spike'].values[0])
            offset_tc_all.append(tc.values[0])

            offset_r_all.append(weighted_circmean(tc.angle, tc.sel(unit=0).values)[1])


        offset_si_all = pd.Series(offset_si_all, index=offset_values)
        
        offset_tc_all = pd.DataFrame(np.array(offset_tc_all), index=offset_values)

        offset_r_all = pd.Series(offset_r_all, index=offset_values)


        offset_si[s] = offset_si_all
        offset_r[s] = offset_r_all


offset_si_all = pd.concat(offset_si, axis=1)
offset_r_all = pd.concat(offset_r, axis=1)
# offset_si_all = offset_si_all - offset_si_all.mean(0)
# offset_si_all = offset_si_all / offset_si_all.std(0)

figure()
subplot(2,1,1)
plot(offset_si_all.mean(1), color='k', linewidth=2)
subplot(212)
plot(offset_r_all.mean(1), color='k', linewidth=2)
show()
