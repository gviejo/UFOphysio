# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-09 14:15:58
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-27 17:32:11
import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
import pynapple as nap
# import pynacollada as pyna
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import hilbert

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


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_power(fp, timestep, chs, freq_band, frequency=20000, wsize=41):
    """
    """
    batch_size = frequency*100
    starts = np.arange(0, len(timestep), batch_size)

    all_batch = np.zeros(len(timestep))

    for i,s in enumerate(starts):
        print(i/len(starts), end="\r", flush=True)

        meanSS = np.zeros(np.minimum(batch_size,len(timestep)-s))

        sig = fp[s:s+batch_size,chs]
        ts = timestep[s:s+batch_size]

        for j in range(sig.shape[1]):            
            b, a = _butter_bandpass(freq_band[0], freq_band[1], frequency, order=2)            
            power = np.abs(hilbert(filtfilt(b, a, sig[:,j])))
            window = np.ones(wsize)/wsize            
            SS = filtfilt(window, 1, power)
            meanSS += SS 
        meanSS = meanSS / len(chs)
        
        all_batch[s:s+batch_size] = meanSS
                
    return nap.Tsd(t=timestep, d=all_batch)

def detect_dentate_spikes(fp, channels, timestep, frequency=1250):

    freq_band = (1, 100)
    wsize = 41
    thres_band = (3.5, 100)
    duration_band = (15, 100) # in ms
    min_inter_duration = 50 # in ms
    
    meanSS =  compute_power(fp, timestep, [channels[0]], freq_band, frequency=frequency, wsize=wsize)
    meanctr = compute_power(fp, timestep, [channels[1]], freq_band, frequency=frequency, wsize=wsize)

    SS = meanSS.d - meanctr.d
    nSS = (SS - np.mean(SS))/np.std(SS)
    nSS = nap.Tsd(t=meanSS.t, d=nSS)

    ds_tsd = []
    ds_ep = []

    # Round1 : Detecting Oscillation Periods by thresholding normalized signal
    try:
        nSS2 = nSS.threshold(thres_band[0], method='above')
    except:
        nSS2 = None

    if nSS2 is not None:
        nSS3 = nSS2.threshold(thres_band[1], method='below')

        # Round 2 : Excluding oscillation whose length < min_duration and greater than max_duration
        osc_ep = nSS3.time_support
        osc_ep = osc_ep.drop_short_intervals(duration_band[0], time_units = 'ms')
        osc_ep = osc_ep.drop_long_intervals(duration_band[1], time_units = 'ms')

        # Round 3 : Merging oscillation if inter-oscillation period is too short
        osc_ep = osc_ep.merge_close_intervals(min_inter_duration, time_units = 'ms')
        # osc_ep = osc_ep.reset_index(drop=True)

        # Extracting Oscillation peak
        osc_max = []
        osc_tsd = []
        for i in osc_ep.index:
            tmp = nSS.get(osc_ep[i,0], osc_ep[i,1])
            osc_tsd.append(tmp.index[np.argmax(tmp)])
            osc_max.append(np.max(tmp))

        osc_max = np.array(osc_max)
        osc_tsd = np.array(osc_tsd)

        osc_tsd = nap.Tsd(t=osc_tsd, d=osc_max)

        ds_tsd = osc_tsd
        ds_ep =osc_ep

        return ds_ep, ds_tsd, nSS
    else:
        return nap.IntervalSet([], []), nap.Tsd([], []), nSS



