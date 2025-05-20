#!/usr/bin/env python
'''

'''
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec




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

datasets = np.unique(datasets)
# datasets = np.genfromtxt('/media/guillaume/LaCie/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')


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

allr = []
pearson = {}

for s in datasets:
    print(s)
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
    idx = spikes.index[spikes.metadata["location"].str.contains("adn|lmn").values]
    spikes = spikes[idx]
    
    ufo_ep, ufo_ts = loadUFOs(path)
    
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)
    
    # CHECKING HALF EPOCHS
    wake2_ep = splitWake(position.time_support.loc[[0]])    
    tokeep2 = []
    stats2 = []
    tcurves2 = []   
    for i in range(2):
        tcurves_half = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tcurves_half = smoothAngularTuningCurves(tcurves_half)

        tokeep, stat = findHDCells(tcurves_half)
        tokeep2.append(tokeep)
        stats2.append(stat)
        tcurves2.append(tcurves_half)       
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
    

    spikes = spikes[tokeep]
    groups = spikes._metadata.loc[tokeep].groupby("location").groups

    tcurves         = tuning_curves[tokeep]
    peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 
    
    sws_ep = sws_ep.set_diff(ufo_ep)
    
    ############################################################################################### 
    # PEARSON CORRELATION
    ###############################################################################################
    rates = {}
    for e, ep, bin_size, std in zip(['wak', 'sws', 'ufo'], [newwake_ep, sws_ep, sws_ep], [0.1, 0.02, 0.001], [3, 3, 3]):
        count = spikes.count(bin_size, ep)
        rate = count/bin_size
        rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        rate = zscore_rate(rate)
        rates[e] = rate 

    from itertools import product
    pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
    pairs = pd.MultiIndex.from_tuples(pairs, names=['first', 'second'])
    r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)
    for p in r.index:
        for ep in rates.keys():
            r.loc[p, ep] = scipy.stats.pearsonr(rates[ep][int(p[0])],rates[ep][int(p[1])])[0]

    name = data.basename
    pairs = list(product([name+'_'+str(n) for n in groups['adn']], [name+'_'+str(n) for n in groups['lmn']]))
    pairs = pd.MultiIndex.from_tuples(pairs)
    r.index = pairs

    #######################
    # COMPUTING PEARSON R FOR EACH SESSION
    #######################
    pearson[s] = np.zeros((3))
    pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['rem'])[0]
    pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
    pearson[s][2] = len(spikes)

    #######################
    # SAVING
    #######################
    allr.append(r)

allr = pd.concat(allr, 0)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['rem', 'sws', 'count']

datatosave = {
    'allr':allr,
    'pearsonr':pearson
    }

# datatosave = {'allr':allr}
# cPickle.dump(datatosave, open(os.path.join('../data/', 'All_correlation_ADN_LMN.pickle'), 'wb'))
cPickle.dump(datatosave, open(os.path.join('/home/guillaume/Dropbox/CosyneData', 'All_correlation_ADN_LMN.pickle'), 'wb'))


figure()
subplot(131)
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
title('r = '+str(np.round(r, 3)))

subplot(132)
plot(allr['wak'], allr['rem'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('rem')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
title('r = '+str(np.round(r, 3)))

show()

