# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   gviejo
# @Last Modified time: 2025-01-08 15:26:30

# %%
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore

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

# %%
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
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

datasets = np.hstack([
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)


allr = []
pearson = {}

for s in datasets:
# for s in ["LMN-ADN/A5011/A5011-201014A"]:
# for s in ['LMN-ADN/A5021/A5021-210521A']:
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    
    if os.path.exists(filepath):
        
        nwb = nap.load_file(filepath)
        
        spikes = nwb['units']
        spikes = spikes.getby_threshold("rate", 1)

        position = []
        columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        for k in columns:
            position.append(nwb[k].values)
        position = np.array(position)
        position = np.transpose(position)
        position = nap.TsdFrame(
            t=nwb['x'].t,
            d=position,
            columns=columns,
            time_support=nwb['position_time_support'])

        epochs = nwb['epochs']
        wake_ep = epochs[epochs.tags == "wake"]
        sws_ep = nwb['sws']
        rem_ep = nwb['rem']
        
        ufo_ep, ufo_ts = loadUFOs(path)
        
        
        # hmm_eps = []
        # try:
        #     filepath = os.path.join(data_directory, s, os.path.basename(s))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep0.npz"))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep1.npz"))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep2.npz"))
        # except:
        #     pass

        spikes = spikes[spikes.location == "adn"]
    
        if len(spikes):
            ############################################################################################### 
            # COMPUTING TUNING CURVES
            ###############################################################################################
            tuning_curves = nap.compute_1d_tuning_curves(
                spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
            
            SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
            spikes.set_info(SI)

            spikes = spikes[spikes.SI>0.1]

            # CHECKING HALF EPOCHS
            wake2_ep = splitWake(position.time_support.loc[[0]])    
            tokeep2 = []
            stats2 = []
            tcurves2 = []   
            for i in range(2):
                tcurves_half = nap.compute_1d_tuning_curves(
                    spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                    ep = wake2_ep[i]
                    )
                tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

                tokeep, stat = findHDCells(tcurves_half)
                tokeep2.append(tokeep)
                stats2.append(stat)
                tcurves2.append(tcurves_half)       
            tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
            

            if len(tokeep) > 5 and ufo_ep is not None:
                print(s)

                spikes = spikes[tokeep]
                # groups = spikes._metadata.loc[tokeep].groupby("location").groups
                tcurves         = tuning_curves[tokeep]

                try:
                    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
                except:
                    velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
                
                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################
                rates = {}
                for e, ep, bin_size, std in zip(['wak', 'sws', 'ufo'], [newwake_ep, sws_ep, sws_ep], [0.1, 0.02, 0.002], [3, 3, 3]):
                    ep = ep.drop_short_intervals(bin_size*22)
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size
                    # rate = rate.as_dataframe()
                    rate = rate.smooth(std=bin_size*std, windowsize=bin_size*20).as_dataframe()
                    rate = rate.apply(zscore)            
                    rates[e] = nap.TsdFrame(rate)

                rates['ufo'] = rates['ufo'].restrict(ufo_ep)
                
                # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
                pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
                pairs = pd.MultiIndex.from_tuples(pairs)
                r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                for ep in rates.keys():
                    tmp = np.corrcoef(rates[ep].values.T)
                    r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

                name = basename    
                pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
                pairs = pd.MultiIndex.from_tuples(pairs)
                r.index = pairs

                #######################
                # Angular differences
                #######################
                peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))                
                for p, (i, j) in zip(pairs, list(combinations(spikes.keys(), 2))):
                    r.loc[p, 'ang'] = min(np.abs(peaks[i] - peaks[j]), 2*np.pi-np.abs(peaks[i] - peaks[j]))

                
                #######################
                # COMPUTING PEARSON R FOR EACH SESSION
                #######################
                pearson[s] = np.zeros((6))*np.nan
                pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['ufo'])[0]
                pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
                # if len(hmm_eps):
                #     for i in range(len(hmm_eps)):
                #         pearson[s][i+2] = scipy.stats.pearsonr(r['wak'], r["ep{}".format(i)])[0]

                pearson[s][-1] = len(spikes)

                #######################
                # SAVING
                #######################
                allr.append(r)

                # sys.exit()

allr = pd.concat(allr)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['ufo', 'sws', 'ep0', 'ep1', 'ep2', 'count']

# datatosave = {
#     'allr':allr,
#     'pearsonr':pearson
#     }
    
# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'wb'))

#%%
figure()
for i, e in enumerate(["ufo", "sws"]):
    subplot(1, 3, i+1)
    tmp = allr[['wak', e]].dropna()
    plot(tmp['wak'], tmp[e], 'o', color = 'red', alpha = 0.5)
    m, b = np.polyfit(tmp['wak'].values, tmp[e].values, 1)
    x = np.linspace(tmp['wak'].min(), tmp['wak'].max(),5)
    plot(x, x*m + b)
    xlabel('wake')
    ylabel(e)
    
    r, p = scipy.stats.pearsonr(tmp['wak'], tmp[e])
    title('r = '+str(np.round(r, 3))+f'\n p={p}')

subplot(133)
plot(np.zeros(len(pearson))+np.random.randn(len(pearson))*0.1, pearson['ufo'].values, 'o')
plot(np.ones(len(pearson))+np.random.randn(len(pearson))*0.1, pearson['sws'].values, 'o') 
plot([-0.1, 0.1], [pearson['ufo'].mean(), pearson['ufo'].mean()], linewidth=3)
plot([0.9, 1.1], [pearson['sws'].mean(), pearson['sws'].mean()], linewidth=3)
print(scipy.stats.wilcoxon(pearson["ufo"], pearson["sws"]))
print(scipy.stats.ttest_ind(pearson["ufo"], pearson["sws"]))

xticks([0,1], ['ufo', 'sws'])
ylim(-1, 1)

show()








# %%s