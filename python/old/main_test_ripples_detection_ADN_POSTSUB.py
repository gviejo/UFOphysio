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



for s in datasets[0:-1]:
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
	tuning_curves 		= smoothAngularTuningCurves(tuning_curves, 10, 2)
	tokeep, stat 		= findHDCells(tuning_curves)

	############################################################
	# PARAMETERS
	############################################################
	frequency = 1250.0
	low_cut = 80
	high_cut = 300
	windowLength = 51
	low_thresFactor = 1.25
	minRipLen = 20 # ms
	maxRipLen = 200 # ms
	minInterRippleInterval = 20 # ms
	limit_peak = 100

	############################################################
	# LOAD LFP
	############################################################
	channels 	= shank_to_channel[5]
	lfp_file 	= os.path.join(path,name+'.eeg')
	lfp 		= loadLFP(lfp_file, n_channels, 0, 1250, 'int16') # Bad but no choice
	nSS 		= zeros(len(lfp))
	for ch in channels:
		lfp 		= loadLFP(lfp_file, n_channels, ch, 1250, 'int16')
		signal = butter_bandpass_filter(lfp.values, low_cut, high_cut, frequency, order = 4)
		squared_signal = np.square(signal)
		window = np.ones(windowLength)/windowLength
		tmp = scipy.signal.filtfilt(window, 1, squared_signal)
		nSS += tmp
	nSS /= len(channels)
	# Removing point above 100000
	nSS = pd.Series(index = lfp.index.values, data = nSS)
	nSS = nSS[nSS<40000]
	nSS = (nSS - np.mean(nSS))/np.std(nSS)
	signal = pd.Series(index = lfp.index.values, data = signal)


	# a = int(17137.788*1e6)
	# b = a + 1e6

	# figure()
	# ax = subplot(211)
	# plot(nSS)
	# axvline(a)
	# subplot(212,sharex = ax)
	# plot(lfp)
	# show()

	######################################################l##################################
	# Round1 : Detecting Ripple Periods by thresholding normalized signal
	thresholded = np.where(nSS > low_thresFactor, 1,0)
	start = np.where(np.diff(thresholded) > 0)[0]
	stop = np.where(np.diff(thresholded) < 0)[0]
	if len(stop) == len(start)-1:
		start = start[0:-1]
	if len(stop)-1 == len(start):
		stop = stop[1:]
	################################################################################################
	# Round 2 : Excluding ripples whose length < minRipLen and greater than Maximum Ripple Length
	if len(start):
		l = (nSS.index.values[stop] - nSS.index.values[start])/1000 # from us to ms
		idx = np.logical_and(l > minRipLen, l < maxRipLen)
	else:	
		print("Detection by threshold failed!")
		sys.exit()

	rip_ep = nts.IntervalSet(start = nSS.index.values[start[idx]], end = nSS.index.values[stop[idx]])

	####################################################################################################################
	# Round 3 : Merging ripples if inter-ripple period is too short
	rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval/1000, time_units = 's')


	#####################################################################################################################
	# Round 4: Discard Ripples with a peak power < high_thresFactor and > limit_peak
	rip_max = []
	rip_tsd = []
	for s, e in rip_ep.values:
		tmp = nSS.loc[s:e]
		rip_tsd.append(tmp.idxmax())
		rip_max.append(tmp.max())

	rip_max = np.array(rip_max)
	rip_tsd = np.array(rip_tsd)

	tokeep = np.logical_and(rip_max > low_thresFactor, rip_max < limit_peak)

	rip_ep = rip_ep[tokeep].reset_index(drop=True)
	rip_tsd = nts.Tsd(t = rip_tsd[tokeep], d = rip_max[tokeep])


	# # t1, t2 = (6002729000,6003713000)
	# t1, t2 = (6002700000,6003713000)
	# figure()
	# ax = subplot(211)
	# plot(lfp.loc[t1:t2])
	# plot(signal.loc[t1:t2])
	# plot(lfp.restrict(rip_ep).loc[t1:t2], '.')
	# subplot(212,sharex = ax)
	# plot(nSS.loc[t1:t2])
	# axhline(low_thresFactor)
	# show()
	
	###########################################################################################################
	# Writing for neuroscope
	###########################################################################################################
	# sws2_ep = sws_ep.merge_close_intervals(60, time_units='s')
	rip_ep			= sws_ep.intersect(rip_ep)
	rip_tsd 		= rip_tsd.restrict(rip_ep)
	if len(rip_tsd)<len(rip_ep):
		todrop = []
		for i in rip_ep.index:
			a = rip_tsd.restrict(rip_ep.loc[[i]])
			if len(a)==0:
				todrop.append(i)
	rip_ep = rip_ep.drop(todrop).reset_index(drop=True)



	start = rip_ep.as_units('ms')['start'].values
	peaks = rip_tsd.as_units('ms').index.values
	ends = rip_ep.as_units('ms')['end'].values

	datatowrite = np.vstack((start,peaks,ends)).T.flatten()

	n = len(rip_ep)

	texttowrite = np.vstack(((np.repeat(np.array(['PyRip start 1']), n)), 
							(np.repeat(np.array(['PyRip peak 1']), n)),
							(np.repeat(np.array(['PyRip stop 1']), n))
								)).T.flatten()

	evt_file = os.path.join(path,name+'.evt.py.rip')
	f = open(evt_file, 'w')
	for t, n in zip(datatowrite, texttowrite):
		f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
	f.close()		






