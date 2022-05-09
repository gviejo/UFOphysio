import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from neurodsp.burst import detect_bursts_dual_threshold

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_ADN_POS.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

hd_adn = []
hd_pos = []
nhd_pos = []
allcc_rip = []
allcc_ufo = []
allfr = []

for s in datasets[1:]:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))

	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)	
	sleep_ep		= loadEpoch(path, 'sleep')	
	sws_ep 			= loadEpoch(path, 'sws')
	
	tuning_curves 		= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)	
	tuning_curves 		= smoothAngularTuningCurves(tuning_curves, 20, 2)
	tokeep, stat 		= findHDCells(tuning_curves)

	neurons 			= [name+'_'+str(n) for n in spikes.keys()]

	rip_ep, rip_tsd 	= loadRipples(path)
	ufo_ep, ufo_tsd 	= loadUFOs(path)
	# taking start of ufos
	# ufo_tsd = nts.Ts(t = ufo_ep['start'].values)

	# CROSS CORR RIP UFO
	cc_rip_ufo = compute_EventCrossCorr({0:ufo_tsd}, rip_tsd, sws_ep, binsize=5, nbins=400, norm=True)

	#######################
	# RIPPLES CROSS-CORR
	#######################	
	cc_rip = compute_EventCrossCorr(spikes, rip_tsd, sws_ep, binsize = 5, nbins = 200, norm=True)
	cc_rip.columns = neurons

	cc_fast = cc_rip.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)	
	cc_slow = cc_rip.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)	

	cc_rip = (cc_fast - cc_slow)/cc_fast

	#######################
	# UFOS CROSS-CORR
	#######################	
	cc_ufo = compute_EventCrossCorr(spikes, ufo_tsd, sws_ep, binsize = 5, nbins = 200, norm=True)
	cc_ufo.columns = neurons

	cc_fast = cc_ufo.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)	
	cc_slow = cc_ufo.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)	

	cc_ufo = (cc_fast - cc_slow)/cc_fast

	#######################
	# SAVINGS
	#######################	
	fr = computeMeanFiringRate(spikes, [sws_ep, wake_ep], ['sws', 'wake'])
	fr.index = neurons
	neurons = np.array(neurons)

	allcc_rip.append(cc_rip)
	allcc_ufo.append(cc_ufo)
	allfr.append(fr)
	hd_adn.append(neurons[tokeep[shank[tokeep]>7]])
	hd_pos.append(neurons[tokeep[shank[tokeep]<7]])
	nhd_pos.append(neurons[np.arange(len(spikes))[np.logical_and(shank>3, shank<7)]])

	# hd_adn = tokeep[shank[tokeep]>7]
	# hd_pos = tokeep[shank[tokeep]<7]
	# nhd_pos = np.arange(len(spikes))[np.logical_and(shank>3, shank<7)]

	# sys.exit()

hd_adn =  np.hstack(hd_adn)
hd_pos =  np.hstack(hd_pos)
nhd_pos = np.hstack(nhd_pos)
others = [n for n in cc_rip.columns if n not in np.hstack((hd_adn, hd_pos, nhd_pos))]
cc_rip = pd.concat(allcc_rip, 1)
cc_ufo = pd.concat(allcc_ufo, 1)
fr = pd.concat(allfr)
sys.exit()


figure()
subplot(241)
plot(cc_rip[hd_adn], color = 'grey', alpha = 0.4)
plot(cc_rip[hd_adn].mean(1), color = 'red', linewidth = 2)
title('ADN')
subplot(242)
plot(cc_rip[hd_pos], color = 'grey', alpha = 0.4)
plot(cc_rip[hd_pos].mean(1), color = 'blue', linewidth = 2)
title('HD POS')
subplot(243)
plot(cc_rip[nhd_pos], color = 'grey', alpha = 0.4)
plot(cc_rip[nhd_pos].mean(1), color = 'green', linewidth = 2)
title('NHD POS')
subplot(244)
plot(cc_rip[others], color = 'grey', alpha = 0.4)
plot(cc_rip[others].mean(1), color = 'black', linewidth = 2)
title('Others')
subplot(212)
plot(cc_rip[hd_adn].mean(1), color = 'red', linewidth = 2)
plot(cc_rip[hd_pos].mean(1), color = 'blue', linewidth = 2)
plot(cc_rip[nhd_pos].mean(1), color = 'green', linewidth = 2)

figure()
subplot(241)
plot(cc_ufo[hd_adn], color = 'grey', alpha = 0.4)
plot(cc_ufo[hd_adn].mean(1), color = 'red', linewidth = 2)
title('ADN')
subplot(242)
plot(cc_ufo[hd_pos], color = 'grey', alpha = 0.4)
plot(cc_ufo[hd_pos].mean(1), color = 'blue', linewidth = 2)
title('HD POS')
subplot(243)
plot(cc_ufo[nhd_pos], color = 'grey', alpha = 0.4)
plot(cc_ufo[nhd_pos].mean(1), color = 'green', linewidth = 2)
title('NHD POS')
subplot(244)
plot(cc_ufo[others], color = 'grey', alpha = 0.4)
plot(cc_ufo[others].mean(1), color = 'black', linewidth = 2)
title('Others')
subplot(212)
plot(cc_ufo[hd_adn].mean(1), color = 'red', linewidth = 2)
plot(cc_ufo[hd_pos].mean(1), color = 'blue', linewidth = 2)
plot(cc_ufo[nhd_pos].mean(1), color = 'green', linewidth = 2)