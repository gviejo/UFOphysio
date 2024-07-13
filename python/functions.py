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

def getAllInfos(data_directory, datasets):
    allm = np.unique(["/".join(s.split("/")[0:2]) for s in datasets])
    infos = {}
    for m in allm:      
        path = os.path.join(data_directory, m)
        csv_file = list(filter(lambda x: '.csv' in x, os.listdir(path)))[0]
        infos[m.split('/')[1]] = pd.read_csv(os.path.join(path, csv_file), index_col = 0)
    return infos

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
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
    peak            = pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]))
    new_tcurve      = []
    for p in tcurve.columns:    
        x = tcurve[p].index.values - tcurve[p].index[tcurve[p].index.get_loc(peak[p], method='nearest')]
        x[x<-np.pi] += 2*np.pi
        x[x>np.pi] -= 2*np.pi
        tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
        new_tcurve.append(tmp.values)
    new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
    return new_tcurve

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
