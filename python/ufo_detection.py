# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-09 14:15:58
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-19 18:23:08
import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
import pynapple as nap
import pynacollada as pyna
from scipy.signal import filtfilt

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


def detect_ufos(fp, channels, timestep):
    frequency = 20000
    freq_band = (600, 4000)
    thres_band = (5, 100)
    wsize = 61
    duration_band = (1, 10)
    min_inter_duration = 5
    
    batch_size = frequency*500

    ufo_tsd = []
    ufo_ep = []

    starts = np.arange(0, len(timestep), batch_size)

    controls = np.array(list(set(np.arange(fp.shape[1])) - set(channels)))
    #controls = np.random.choice(controls, len(channels), replace=False)

    # idx = np.logical_and(timestep > ep.loc[0, "start"], timestep<ep.loc[0,"end"])

    for i,s in enumerate(starts):        

        meannSS = np.zeros(np.minimum(batch_size,len(timestep)-s))
        for j, c in enumerate(channels):
            print(i/len(starts),j/len(channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
            squared_signal = np.square(signal.values)
            window = np.ones(wsize)/wsize
            nSS = filtfilt(window, 1, squared_signal)
            nSS = (nSS - np.mean(nSS))/np.std(nSS)
            meannSS += nSS        
        meannSS = meannSS / len(channels)        
        
        meanctr = np.zeros(np.minimum(batch_size,len(timestep)-s))  
        for j, c in enumerate(controls):
            print(i/len(starts),j/len(controls), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))            
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
            squared_signal = np.square(signal.values)
            window = np.ones(wsize)/wsize
            nSS = filtfilt(window, 1, squared_signal)
            nSS = (nSS - np.mean(nSS))/np.std(nSS)
            meanctr += nSS        
        meanctr = meanctr / len(controls)
        
        nSS = meannSS - meanctr
        nSS = nap.Tsd(t = timestep[s:s+batch_size], d=nSS)

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
            osc_ep = osc_ep.reset_index(drop=True)

            # Extracting Oscillation peak
            osc_max = []
            osc_tsd = []
            for s, e in osc_ep.values:
                tmp = nSS.loc[s:e]
                osc_tsd.append(tmp.idxmax())
                osc_max.append(tmp.max())

            osc_max = np.array(osc_max)
            osc_tsd = np.array(osc_tsd)

            osc_tsd = nap.Tsd(t=osc_tsd, d=osc_max)

            ufo_tsd.append(osc_tsd.as_series())
            ufo_ep.append(osc_ep.as_units('s'))

    ufo_tsd = pd.concat(ufo_tsd)
    ufo_tsd = nap.Tsd(ufo_tsd)

    ufo_ep = pd.concat(ufo_ep)
    ufo_ep = nap.IntervalSet(ufo_ep)

    return ufo_ep, ufo_tsd

def loadUFOs(path):
    """
    Name of the file should end with .evt.py.ufo
    """
    import os
    name = path.split("/")[-1]
    files = os.listdir(path)
    filename = os.path.join(path, name+'.evt.py.ufo')
    if name+'.evt.py.ufo' in files:
        tmp = np.genfromtxt(path + '/' + name + '.evt.py.ufo')[:,0]
        ripples = tmp.reshape(len(tmp)//3,3)/1000
    else:
        print("No ufo in ", path)
        return None, None
    return (nap.IntervalSet(ripples[:,0], ripples[:,2], time_units = 's'), 
            nap.Ts(ripples[:,1], time_units = 's'))

def loadRipples(path):
    """
    Name of the file should end with .evt.py.ufo
    """
    import os
    name = path.split("/")[-1]
    files = os.listdir(path)
    filename = os.path.join(path, name+'.evt.py.rip')
    if name+'.evt.py.rip' in files:
        tmp = np.genfromtxt(path + '/' + name + '.evt.py.rip')[:,0]
        ripples = tmp.reshape(len(tmp)//3,3)/1000
    else:
        print("No ripples in ", path)
        return None, None
    return (nap.IntervalSet(ripples[:,0], ripples[:,2], time_units = 's'), 
            nap.Ts(ripples[:,1], time_units = 's'))