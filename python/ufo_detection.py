# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-09 14:15:58
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-13 12:25:25
import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
import pynapple as nap
import pynacollada as pyna
from scipy.signal import filtfilt
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

def compute_meanNSS(fp, sign_channels, ctrl_channels, timestep):

    frequency = 20000
    freq_band = (500, 1000)
    wsize = 41

    # # tocut = np.searchsorted(timestep, [t-1, t+1])
    # # fp = fp[tocut[0]:tocut[1]]
    # # timestep = timestep[tocut[0]:tocut[1]]


    # window = np.ones(wsize)/wsize

    # meanpower = np.zeros(len(fp))
    # for j, c in enumerate(sign_channels):        
    #     lfp = nap.Tsd(t=timestep, d = np.array(fp[:,c][:]))
    #     signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
    #     power = np.abs(hilbert(signal.d))        
    #     filt_power = filtfilt(window, 1, power)
    #     meanpower+=filt_power
    # meanpower /= len(sign_channels)

    # meanctr = np.zeros(len(lfp))
    # for j, c in enumerate(ctrl_channels):
    #     lfp = nap.Tsd(t=timestep, d = np.array(fp[:,c][:]))        
    #     signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
    #     power = np.abs(hilbert(signal.d))
    #     filt_power = filtfilt(window, 1, power)
    #     meanctr += filt_power
    # meanctr /= len(ctrl_channels)

    # SS = meanpower - meanctr
    # nSS = (SS - np.mean(SS))/np.std(SS)


    # ax = subplot(211)
    # plot(meanpower)
    # plot(meanctr)
    # subplot(212, sharex=ax)
    # plot(nSS)
    # show()

    
    batch_size = frequency*36000

    starts = np.arange(0, len(timestep), batch_size)

    allnSS = []

    for i,s in enumerate(starts):

        meanSS = np.zeros(np.minimum(batch_size,len(timestep)-s))
        for j, c in enumerate(sign_channels):
            print(i/len(starts),j/len(sign_channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)            
            power = np.abs(hilbert(signal.d))
            window = np.ones(wsize)/wsize            
            SS = filtfilt(window, 1, power)
            meanSS += SS 
        meanSS = meanSS / len(sign_channels)        
        
        meanctr = np.zeros(np.minimum(batch_size,len(timestep)-s))  
        for j, c in enumerate(ctrl_channels):
            print(i/len(starts),j/len(ctrl_channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))            
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
            power = np.abs(hilbert(signal.d))
            window = np.ones(wsize)/wsize
            SS = filtfilt(window, 1, power)
            meanctr += SS
        meanctr = meanctr / len(ctrl_channels)
        
        SS = meanSS - meanctr
        nSS = (SS - np.mean(SS))/np.std(SS)

        nSS = nap.Tsd(t = timestep[s:s+batch_size], d=nSS)

        allnSS.append(nSS)
        
    return np.hstack(allnSS)

def detect_ufos(fp, sign_channels, ctrl_channels, timestep):
    frequency = 20000
    freq_band = (600, 2000)
    thres_band = (3, 100)
    wsize = 101
    duration_band = (3, 30)
    min_inter_duration = 5
    
    batch_size = frequency*500

    ufo_tsd = []
    ufo_ep = []

    starts = np.arange(0, len(timestep), batch_size)

    # controls = np.array(list(set(np.arange(fp.shape[1])) - set(channels)))
    # controls = np.random.choice(controls, len(channels), replace=False)

    # idx = np.logical_and(timestep > ep.loc[0, "start"], timestep<ep.loc[0,"end"])

    for i,s in enumerate(starts):

        meannSS = np.zeros(np.minimum(batch_size,len(timestep)-s))
        for j, c in enumerate(sign_channels):
            print(i/len(starts),j/len(sign_channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
            squared_signal = np.square(signal.values)
            window = np.ones(wsize)/wsize
            nSS = filtfilt(window, 1, squared_signal)
            nSS = nSS - np.mean(nSS)
            nSS = nSS/np.std(nSS)
            meannSS += nSS        
        meannSS = meannSS / len(sign_channels)        
        
        meanctr = np.zeros(np.minimum(batch_size,len(timestep)-s))  
        for j, c in enumerate(ctrl_channels):
            print(i/len(starts),j/len(ctrl_channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))            
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
            squared_signal = np.square(signal.values)
            window = np.ones(wsize)/wsize
            nSS = filtfilt(window, 1, squared_signal)
            nSS = nSS - np.mean(nSS)
            nSS = nSS/np.std(nSS)
            meanctr += nSS        
        meanctr = meanctr / len(ctrl_channels)
        
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
            for i in osc_ep.index.values:
                tmp = nSS.restrict(osc_ep.loc[[i]])
                osc_tsd.append(tmp.index[np.argmax(tmp)])
                osc_max.append(np.max(tmp))

            osc_max = np.array(osc_max)
            osc_tsd = np.array(osc_tsd)

            osc_tsd = pd.Series(index=osc_tsd, data=osc_max)

            ufo_tsd.append(osc_tsd)
            ufo_ep.append(osc_ep.as_units('s'))

    ufo_tsd = pd.concat(ufo_tsd)
    ufo_tsd = nap.Tsd(ufo_tsd)

    ufo_ep = pd.concat(ufo_ep)
    ufo_ep = nap.IntervalSet(ufo_ep)

    return ufo_ep, ufo_tsd

def detect_ufos_v2(fp, sign_channels, ctrl_channels, timestep):

    nSS = compute_meanNSS(fp, sign_channels, ctrl_channels, timestep)

    thres_band = (3, 100)    
    duration_band = (2, 30)
    min_inter_duration = 1
    

    ufo_tsd = []
    ufo_ep = []

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
            tmp = nSS.restrict(osc_ep[i])
            osc_tsd.append(tmp.index[np.argmax(tmp)])
            osc_max.append(np.max(tmp))

        osc_max = np.array(osc_max)
        osc_tsd = np.array(osc_tsd)

        osc_tsd = nap.Tsd(t=osc_tsd, d=osc_max)

        ufo_tsd = osc_tsd
        ufo_ep =osc_ep

        return ufo_ep, ufo_tsd, nSS
    else:
        return nap.IntervalSet([], []), nap.Tsd([], []), nSS

def detect_ufos_v3(fp, sign_channels, ctrl_channels, timestep, clu, res):
    frequency = 20000
    freq_band = (600, 2000)
    thres_band = (3, 100)
    wsize = 101
    duration_band = (2, 30)
    min_inter_duration = 5
    
    batch_size = frequency*600

    ufo_tsd = []
    ufo_ep = []

    starts = np.arange(0, len(timestep), batch_size)

    # controls = np.array(list(set(np.arange(fp.shape[1])) - set(channels)))
    # controls = np.random.choice(controls, len(channels), replace=False)

    # idx = np.logical_and(timestep > ep.loc[0, "start"], timestep<ep.loc[0,"end"])

    for i,s in enumerate(starts):

        meannSS = np.zeros(np.minimum(batch_size,len(timestep)-s))

        # get spike times in batch
        idx = np.searchsorted(res, np.array([starts[i], starts[i]+batch_size]))
        res_in_batch = res[idx[0]:idx[1]]

        for j, c in enumerate(sign_channels):
            print(i/len(starts),j/len(sign_channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))

            # Removing the spikes by interpolation
            ilfp = remove_spikes_with_interp(lfp, res_in_batch, frequency)

            # tsg = nap.Tsd(t=res[res<s+batch_size]/20000, d=clu[res<s+batch_size]).to_tsgroup()
            # wavef = nap.compute_event_trigger_average(tsg, signal, 1/20000, (-0.001, 0.002))

            signal = pyna.eeg_processing.bandpass_filter(ilfp, freq_band[0], freq_band[1], frequency)
            power = np.abs(hilbert(signal.d))
            window = np.ones(wsize)/wsize
            nSS = filtfilt(window, 1, power)
            nSS = nSS - np.mean(nSS)
            nSS = nSS/np.std(nSS)
            meannSS += nSS        
        meannSS = meannSS / len(sign_channels)        
        
        meanctr = np.zeros(np.minimum(batch_size,len(timestep)-s))  
        for j, c in enumerate(ctrl_channels):
            print(i/len(starts),j/len(ctrl_channels), end="\r", flush=True)
            lfp = nap.Tsd(t=timestep[s:s+batch_size], d = np.array(fp[s:s+batch_size,c][:]))
            signal = pyna.eeg_processing.bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
            power = np.abs(hilbert(signal.d))
            window = np.ones(wsize)/wsize
            nSS = filtfilt(window, 1, power)
            nSS = nSS - np.mean(nSS)
            nSS = nSS/np.std(nSS)
            meanctr += nSS        
        meanctr = meanctr / len(ctrl_channels)
        
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
            # osc_ep = osc_ep.reset_index(drop=True)

            # Extracting Oscillation peak
            osc_max = []
            osc_tsd = []
            for i in range(len(osc_ep)):
                tmp = nSS.restrict(osc_ep[i])
                osc_tsd.append(tmp.index[np.argmax(tmp)])
                osc_max.append(np.max(tmp))

            osc_max = np.array(osc_max)
            osc_tsd = np.array(osc_tsd)

            osc_tsd = pd.Series(index=osc_tsd, data=osc_max)

            ufo_tsd.append(osc_tsd)
            ufo_ep.append(osc_ep.as_units('s'))

    ufo_tsd = pd.concat(ufo_tsd)
    ufo_tsd = nap.Tsd(ufo_tsd)

    ufo_ep = pd.concat(ufo_ep)
    ufo_ep = nap.IntervalSet(ufo_ep)

    return ufo_ep, ufo_tsd

def remove_spikes_with_interp(lfp, res_in_batch, frequency):
    ep2 = lfp.time_support.set_diff(
        nap.IntervalSet(
            start=(res_in_batch-5)/frequency,
            end = (res_in_batch+5)/frequency,
            )
        )
    lfp2 = lfp.restrict(ep2)
    ilfp = lfp2.interpolate(lfp, ep = lfp.time_support)
    return ilfp

def loadUFOs(path):
    """
    Name of the file should end with .evt.py.ufo
    """
    import os
    name = path.split("/")[-1]
    files = os.listdir(path)
    filename = os.path.join(path, name+'.evt.py.ufo')
    # if name+'.evt.py.ufo' in files:
    try:
        tmp = np.genfromtxt(path + '/' + name + '.evt.py.ufo')[:,0]
        ripples = tmp.reshape(len(tmp)//3,3)/1000
        return (nap.IntervalSet(ripples[:,0], ripples[:,2], time_units = 's'), 
                nap.Ts(ripples[:,1], time_units = 's'))    
    except:
        print("No ufo in ", path)
        return None, None

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



# def downsample(tsd, up, down):
#   import scipy.signal

#   dtsd = scipy.signal.resample_poly(tsd.values, up, down)
#   dt = tsd.as_units('s').index.values[np.arange(0, tsd.shape[0], down)]
#   if len(tsd.shape) == 1:     
#       return nap.Tsd(dt, dtsd, time_units = 's')
#   elif len(tsd.shape) == 2:
#       return nap.TsdFrame(dt, dtsd, time_units = 's', columns = list(tsd.columns))

