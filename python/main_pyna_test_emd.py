# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-09 16:24:58
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-06-13 17:37:22
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-09 16:15:44

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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)


for s in ['LMN-ADN/A5002/A5002-200303B']:

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
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]

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

    dummy = nap.Tsd(t=timestep,d=np.arange(0,len(timestep)))

    ep = nap.IntervalSet(start=3181.074,end=3182.074)
    idx = dummy.restrict(ep)

    lfp = fp[idx.values,41][:]


sys.exit()

import emd

imf = emd.sift.sift(lfp)

IP, IF, IA = emd.spectra.frequency_transform(imf, 20000, 'hilbert')

freq_range = (300, 5000, 100, 'log')

f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)


emd.plotting.plot_imfs(imf)

emd.plotting.plot_hilberthuang(hht, idx.index.values, f,
                               log_y=True)


# Create a signal with a dynamic oscillation
sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds
time_vect = np.linspace(0, seconds, num_samples)

z = emd.simulate.ar_oscillator(25, sample_rate, seconds, r=0.975)[:, 0]
imf = emd.sift.sift(z)
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')

freq_range = (0.1, 50, 48)
hht_f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, mode='amplitude', sum_time=False)
hht = scipy.ndimage.gaussian_filter(hht, 1)
nperseg = 2048
ftf, ftt, ftZ = signal.stft(z, nperseg=nperseg, fs=sample_rate, noverlap=nperseg-1)
plt.figure(figsize=(8, 10))
plt.subplot(311)
plt.plot(time_vect, z)
plt.xlim(1, 9)
plt.title('Signal')
plt.subplot(312)
plt.pcolormesh(ftt, ftf, np.abs(ftZ), cmap='hot_r')
plt.ylim(0, 50)
plt.xlim(1, 9)
plt.title('Short-Time Fourier-Transform')
plt.ylabel('Frequency (Hz)')
plt.subplot(313)
plt.pcolormesh(time_vect, hht_f, hht, cmap='hot_r')
plt.ylim(0, 50)
plt.title('Hilbert-Huang Transform')
plt.xlim(1, 9)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')