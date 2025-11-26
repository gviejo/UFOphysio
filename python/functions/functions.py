# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-02-28 16:16:36
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-06-04 17:06:14
import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
from itertools import combinations
from pycircstat.descriptive import mean as circmean
from pylab import *
import pynapple as nap
from matplotlib.colors import hsv_to_rgb
# import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import xarray as xr

def getAllInfos(data_directory, datasets):
    allm = np.unique(["/".join(s.split("/")[0:2]) for s in datasets])
    infos = {}
    for m in allm:      
        path = os.path.join(data_directory, m)
        csv_file = list(filter(lambda x: '.csv' in x, os.listdir(path)))[0]
        infos[m.split('/')[1]] = pd.read_csv(os.path.join(path, csv_file), index_col = 0)
    return infos

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
    new_tuning_curves = tuning_curves.copy(deep=True)
    tmp = []
    for i in tuning_curves.unit:
        tcurves = tuning_curves.sel(unit=i)
        index = tcurves.coords[tcurves.dims[0]].values
        offset = np.mean(np.diff(index))
        padded  = pd.Series(index = np.hstack((index-(2*np.pi)-offset,
                                                index,
                                                index+(2*np.pi)+offset)),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)
        tmp.append(smoothed.loc[index])

    new_tuning_curves.values = np.array(tmp)

    return new_tuning_curves


def smoothAngularTuningCurves_old(tuning_curves, window = 20, deviation = 3.0):
    new_tuning_curves = {}  
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        offset = np.mean(np.diff(tcurves.index.values))
        padded  = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi)+offset)),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)      
        new_tuning_curves[i] = smoothed.loc[tcurves.index]

    new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

    return new_tuning_curves

def splitWake(ep):
    if len(ep) != 1:
        print('Cant split wake in 2')
        sys.exit()
    tmp = np.zeros((2,2))
    tmp[0,0] = ep.values[0,0]
    tmp[1,1] = ep.values[0,1]
    tmp[0,1] = tmp[1,0] = ep.values[0,0] + np.diff(ep.values[0])/2
    return nap.IntervalSet(start = tmp[:,0], end = tmp[:,1])

def zscore_rate(rate):
    time_support = rate.time_support
    idx = rate.index
    cols = rate.columns
    rate = rate.values
    rate = rate - rate.mean(0)
    rate = rate / rate.std(0)
    rate = nap.TsdFrame(t=idx.values, d=rate, time_support = time_support, columns = cols)
    return rate

def findHDCells(tuning_curves, z = 50, p = 0.0001 , m = 1):
    """
        Peak firing rate larger than 1
        and Rayleigh test p<0.001 & z > 100
    """
    cond1 = tuning_curves.max()>m
    from pycircstat.tests import rayleigh
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
    for k in tuning_curves:
        stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
    cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
    tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
    return tokeep, stat 

def computeLinearVelocity(pos, ep, bin_size):
    pos = pos.restrict(ep)
    pos2 = pos.bin_average(bin_size)
    pos2 = pos2.smooth(1, 100)
    speed = np.sqrt(np.sum(np.power(pos2.values[1:, :] - pos2.values[0:-1, :], 2), 1))
    t = pos2.index.values[0:-1]+np.diff(pos2.index.values)
    speed = nap.Tsd(t = t, d=speed, time_support = ep)
    return speed

def computeAngularVelocity(angle, ep, bin_size):
    """this function only works for single epoch
    """        
    tmp = np.unwrap(angle.restrict(ep).values)
    tmp = pd.Series(index=angle.restrict(ep).index.values, data=tmp)
    tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
    tmp = nap.Tsd(t = tmp.index.values, d = tmp.values)    
    tmp = tmp.bin_average(bin_size)
    t = tmp.index.values[0:-1]+np.diff(tmp.index.values)
    velocity = nap.Tsd(t=t, d = np.diff(tmp))
    return velocity

def getRGB(angle, ep, bin_size):
    angle = angle.restrict(ep)
    bins = np.arange(ep.as_units('s').start.iloc[0], ep.as_units('s').end.iloc[-1]+bin_size, bin_size)  
    tmp = angle.as_series().groupby(np.digitize(angle.as_units('s').index.values, bins)-1).mean()
    tmp2 = pd.Series(index = np.arange(0, len(bins)-1), dtype = float64)
    tmp2.loc[tmp.index.values] = tmp.values
    tmp2 = tmp2.fillna(method='ffill')    
    tmp = nap.Tsd(t = bins[0:-1] + np.diff(bins)/2., d = tmp2.values, time_support = ep)
    H = tmp.values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)
    RGB = nap.TsdFrame(t = tmp.index.values, d = RGB, time_support = ep, columns = ['R', 'G', 'B'])
    return RGB

def getBinnedAngle(angle, ep, bin_size):
    angle = angle.restrict(ep)
    bins = np.arange(ep.as_units('s').start.iloc[0], ep.as_units('s').end.iloc[-1]+bin_size, bin_size)  
    tmp = angle.as_series().groupby(np.digitize(angle.as_units('s').index.values, bins)-1).mean()
    tmp2 = pd.Series(index = np.arange(0, len(bins)-1), dtype = float64)
    tmp2.loc[tmp.index.values] = tmp.values
    tmp2 = tmp2.fillna(method='ffill')
    tmp = nap.Tsd(t = bins[0:-1] + np.diff(bins)/2., d = tmp2.values, time_support = ep)
    return tmp

def xgb_decodage(Xr, Yr, Xt):      
    n_class = 32
    bins = np.linspace(0, 2*np.pi, n_class+1)
    labels = np.digitize(Yr.values, bins)-1
    dtrain = xgb.DMatrix(Xr.values, label=labels)
    dtest = xgb.DMatrix(Xt.values)

    params = {'objective': "multi:softprob",
    'eval_metric': "mlogloss", #loglikelihood loss
    'seed': 2925, #for reproducibility    
    'learning_rate': 0.01,
    'min_child_weight': 2, 
    # 'n_estimators': 1000,
    # 'subsample': 0.5,    
    'max_depth': 5, 
    'gamma': 0.5,
    'num_class':n_class}

    num_round = 50
    bst = xgb.train(params, dtrain, num_round)
    
    ymat = bst.predict(dtest)
    pclas = np.argmax(ymat, 1)

    clas = bins[0:-1] + np.diff(bins)/2

    Yp = clas[pclas]

    Yp = nap.Tsd(t = Xt.index.values, d = Yp, time_support = Xt.time_support)
    proba = nap.TsdFrame(t = Xt.index.values, d = ymat, time_support = Xt.time_support)

    return Yp, proba, bst

def xgb_predict(bst, Xt, n_class = 120):
    dtest = xgb.DMatrix(Xt.values)
    ymat = bst.predict(dtest)
    pclas = np.argmax(ymat, 1)    
    bins = np.linspace(0, 2*np.pi, n_class+1)
    clas = bins[0:-1] + np.diff(bins)/2
    Yp = clas[pclas]
    Yp = nap.Tsd(t = Xt.index.values, d = Yp, time_support = Xt.time_support)
    proba = nap.TsdFrame(t = Xt.index.values, d = ymat, time_support = Xt.time_support)
    return Yp, proba


def smoothAngle(angle, std):
    t = angle.index.values
    d = np.unwrap(angle.values)
    tmp = pd.Series(index = t, data = d)
    tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=std)
    tmp = tmp%(2*np.pi)
    return nap.Tsd(tmp, time_support = angle.time_support)

def getAngularVelocity(angle, bin_size = None):
    dv = np.abs(np.diff(np.unwrap(angle.values)))
    dt = np.diff(angle.index.values)
    t = angle.index.values[0:-1]
    idx = np.where(dt<2*bin_size)[0]
    av = nap.Tsd(t=t[idx]+dt[idx]/2, d=dv[idx], time_support = angle.time_support)
    return av, idx

def plot_tc(tuning_curves, spikes):
    shank = spikes._metadata['group']
    figure()
    count = 1
    for j in np.unique(shank):
        neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
        for k,i in enumerate(neurons):
            subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
            plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i))
            legend()
            count+=1
            gca().set_xticklabels([])
    show()

def centerTuningCurves(tcurve):
    """
    center tuning curves by peak
    """
    idxmax = np.argmax(tcurve.values, axis=1)
    shift = tcurve.shape[1]//2 - idxmax
    new_tcurve = np.zeros_like(tcurve.values)
    for i in range(tcurve.shape[0]):
        new_tcurve[i,:] = np.roll(tcurve.values[i,:], shift[i])

    new_index = np.linspace(-np.pi, np.pi, tcurve.shape[1], endpoint=False)
    new_coords = {tcurve.dims[1]: new_index, tcurve.dims[0]: tcurve.coords[tcurve.dims[0]].values}
    return xr.DataArray(data=new_tcurve, coords=new_coords, dims=tcurve.dims)


def compute_ISI_HD(spikes, angle, ep, bins):
    nb_bin_hd = 31
    tc2 = nap.compute_1d_tuning_curves(spikes, angle, nb_bin_hd, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    # angle2 = angle.restrict(ep)
    xbins = np.linspace(0, 2*np.pi, nb_bin_hd)
    xpos = xbins[0:-1] + np.diff(xbins)/2    

    pisiall = {}
    for n in spikes.keys():
        spk = spikes[n]
        isi = nap.Tsd(t = spk.index.values[0:-1]+np.diff(spk.index.values)/2, d=np.diff(spk.index.values))            
        idx = angle.index.get_indexer(isi.index, method="nearest")
        isi_angle = pd.Series(index = angle.index.values, data = np.nan)
        isi_angle.loc[angle.index.values[idx]] = isi.values
        isi_angle = isi_angle.fillna(method='ffill')
        isi_angle = nap.Tsd(isi_angle)        
        isi_angle = isi_angle.restrict(ep)

        # isi_angle = nap.Ts(t = angle.index.values, time_support = ep)
        # isi_angle = isi_angle.value_from(isi, ep)
        
        #data = np.vstack([np.hstack([isi_before.values, isi_after.values]), np.hstack([isi_angle.values, isi_angle.values])])
        data = np.vstack((isi_angle.values, angle.restrict(ep).values))

        pisi, _, _ = np.histogram2d(data[0], data[1], bins=[bins, xbins], weights = np.ones(len(data[0]))/float(len(data[0])))
        m = pisi.max()
        if m>0.0:
            pisi = pisi/m

        pisi = pd.DataFrame(index=bins[0:-1], columns=xpos, data=pisi)
        pisi = pisi.T
        # centering
        offset = tc2[n].idxmax()
        new_index = pisi.index.values - offset
        new_index[new_index<-np.pi] += 2*np.pi
        new_index[new_index>np.pi] -= 2*np.pi
        pisi.index = pd.Index(new_index)
        pisi = pisi.sort_index()
        pisi = pisi.T
        pisiall[n] = pisi

    return pisiall, xbins, bins

def decode_xgb(spikes, eptrain, bin_size_train, eptest, bin_size_test, angle, std = 1):    
    count = spikes.count(bin_size_train, eptrain)
    count = count.as_dataframe()    
    rate_train = count/bin_size_train
    #rate_train = rate_train.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
    rate_train = nap.TsdFrame(rate_train, time_support = eptrain)
    rate_train = zscore_rate(rate_train)
    rate_train = rate_train.restrict(eptrain)
    angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_train).restrict(eptrain)

    count = spikes.count(bin_size_test, eptest)
    count = count.as_dataframe()
    rate_test = count/bin_size_test
    #rate_test = rate_test.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
    rate_test = nap.TsdFrame(rate_test, time_support = eptest)
    rate_test = zscore_rate(rate_test)

    angle_predi, proba, bst = xgb_decodage(Xr=rate_train, Yr=angle2, Xt=rate_test)

    return angle_predi, proba

def loadXML(path):
    """
    path should be the folder session containing the XML file
    Function returns :
        1. the number of channels
        2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
            eeg file first if both are present or both are absent
        3. the mappings shanks to channels as a dict
    Args:
        path : string
    Returns:
        int, int, dict
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, xmlfiles[0])

    from xml.dom import minidom
    xmldoc = minidom.parse(new_path)
    nChannels = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs_eeg = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[
        0].firstChild.data
    if os.path.splitext(xmlfiles[0])[0] + '.dat' in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] + '.eeg' in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    shank_to_keep = {}
    groups = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[
        0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = []
        shank_to_keep[i] = []
        for child in groups[i].getElementsByTagName('channel'):
            shank_to_channel[i].append(int(child.firstChild.data))
            tmp = child.toprettyxml()
            shank_to_keep[i].append(int(tmp[15]))

        # shank_to_channel[i] = np.array([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
        shank_to_channel[i] = np.array(shank_to_channel[i])
        shank_to_keep[i] = np.array(shank_to_keep[i])
        shank_to_keep[i] = shank_to_keep[i] == 0  # ugly
    return int(nChannels), int(fs), shank_to_channel, shank_to_keep

def load_mean_waveforms(path):
    """
    load waveforms
    quick and dirty
    """
    import scipy.io
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, 'Analysis/')
    if os.path.exists(new_path):
        new_path    = os.path.join(path, 'Analysis/')
        files        = os.listdir(new_path)
        if "MeanWaveForms.npz" in files:
            tmp = np.load(os.path.join(new_path, 'MeanWaveForms.npz'), allow_pickle=True)
            meanwavef = tmp['meanwavef'].item()
            maxch = tmp['maxch'].item()
            return meanwavef, maxch

    # Creating /Analysis/ Folder here if not already present
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    files = os.listdir(path)
    clu_files     = np.sort([f for f in files if 'clu' in f and f[0] != '.'])
    spk_files	  = np.sort([f for f in files if 'spk' in f and f[0] != '.'])
    clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2         = np.sort([int(f.split(".")[-1]) for f in spk_files])
    if len(clu_files) != len(spk_files) or not (clu1 == clu2).any():
        print("Not the same number of clu and res files in "+path+"; Exiting ...")
        sys.exit()

    # XML INFO
    n_channels, fs, shank_to_channel, shank_to_keep = loadXML(path)
    from xml.dom import minidom
    xmlfile = os.path.join(path, [f for f in files if f.endswith('.xml')][0])
    xmldoc 		= minidom.parse(xmlfile)
    nSamples 	= int(xmldoc.getElementsByTagName('nSamples')[0].firstChild.data) # assuming constant nSamples

    import xml.etree.ElementTree as ET
    root = ET.parse(xmlfile).getroot()


    count = 0
    meanwavef = {}
    maxch = {}
    for i, s in zip(range(len(clu_files)),clu1):
        clu = np.genfromtxt(os.path.join(path,clu_files[i]),dtype=np.int32)[1:]
        mwf = []
        mch = []
        if np.max(clu)>1:
            # load waveforms
            file = os.path.join(path, spk_files[i])
            f = open(file, 'rb')
            startoffile = f.seek(0, 0)
            endoffile = f.seek(0, 2)
            bytes_size = 2
            n_samples = int((endoffile-startoffile)/bytes_size)
            f.close()
            n_channel = len(root.findall('spikeDetection/channelGroups/group')[s-1].findall('channels')[0])

            data = np.memmap(file, np.int16, 'r', shape = (len(clu), nSamples, n_channel))

            #data = np.fromfile(open(file, 'rb'), np.int16)
            #data = data.reshape(len(clu),nSamples,n_channel)

            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp>1]
            idx_col = np.arange(count, count+len(idx_clu))
            for j,k in zip(idx_clu, idx_col):
                # take only a subsample of spike if too big
                idx = np.sort(np.random.choice(np.where(clu==j)[0], 100))
                meanw = data[idx,:,:].mean(0)
                ch = np.argmax(np.max(np.abs(meanw), 0))
                mwf.append(meanw)
                mch.append(ch)
            mwf = np.array(mwf)
            # mch = pd.Series(index = idx_col, data = mch)
            count += len(idx_clu)
            meanwavef[i] = mwf
            maxch[i] = np.array(mch)

    # meanwavef = pd.concat(meanwavef, axis=1)
    # maxch = pd.concat(maxch)
    # meanwavef.to_hdf(os.path.join(new_path, 'MeanWaveForms.h5'), key='waveforms', mode='w')
    # maxch.to_hdf(os.path.join(new_path, 'MaxWaveForms.h5'), key='channel', mode='w')
    np.savez(os.path.join(new_path, 'MeanWaveForms.npz'), meanwavef=meanwavef, maxch=maxch)
    return meanwavef, maxch