import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

sys.exit()



# for s in datasets[-2:]:
for s in ['A5000/A5002/A5002-200304A']:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	events			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	# wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	# sws_ep 			= loadEpoch(path, 'sws')

	#####################################
	# PARAMETERS
	#####################################
	windowLength = 61
	frequency = 20000
	low_cut = 600
	high_cut = 4000
	nSS_highcut = 50
	low_thresFactor = 6
	high_thresFactor = 100
	minRipLen = 1 # ms
	maxRipLen = 10 # ms
	minInterRippleInterval = 3 # ms
	limit_peak = 100

	if 'A5002' in name:
		datfile = '/mnt/DataGuillaume/LMN-ADN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
	elif 'A1407' in name:
		datfile = '/mnt/DataGuillaume/LMN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
		shank_to_channel = {i+2:shank_to_channel[i] for i in shank_to_channel.keys()}
	elif 'A5001' in name:
		datfile = '/mnt/DataGuillaume/LMN-ADN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
		shank_to_channel = {i-3:shank_to_channel[i] for i in [5, 6, 7, 8]}


	frequency = 20000

	f = open(datfile, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2		
	n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
	duration = n_samples/frequency
	f.close()
	fp = np.memmap(datfile, np.int16, 'r', shape = (n_samples, n_channels))
	timestep = (np.arange(0, n_samples)/frequency)*1e6
	timestep = timestep.astype(np.int)
	duree = len(timestep)	

	dummy = pd.Series(index = timestep, data = 0)

	# # #TO REMOVE
	half_sleep = sleep_ep.loc[[0]]
	duree = len(dummy.loc[half_sleep.start[0]:half_sleep.end[0]].index.values)
	dummy = dummy.loc[half_sleep.start[0]:half_sleep.end[0]]

	# sys.exit()

	nSS = np.zeros((duree,2))

	SS = np.zeros((duree,2))

	for ch in shank_to_channel[2]:
		print(ch)
		##################################################################################################
		# LOADING LFP
		##################################################################################################	
		lfp = pd.Series(index = timestep, data = fp[:,ch])
		# TO REMOVE
		lfp = lfp.loc[half_sleep.start[0]:half_sleep.end[0]]
		#
		
		##################################################################################################
		# FILTERING
		##################################################################################################
		signal			= butter_bandpass_filter(lfp, low_cut, high_cut, frequency, 6)
		squared_signal = np.square(signal)
		window = np.ones(windowLength)/windowLength

		# nSS[:,0] += scipy.ndimage.filters.gaussian_filter1d(squared_signal, 30)
		nSS[:,0] += scipy.signal.filtfilt(window, 1, squared_signal)
		SS[:,0] += squared_signal

		del lfp
		del signal
		del squared_signal

	nSS[:,0] /= len(shank_to_channel[2])
	SS[:,0] /= len(shank_to_channel[2])

	for i, sh in enumerate([3,4,5]):		
		for ch in shank_to_channel[sh]:
			print(ch)
			##################################################################################################
			# LOADING LFP
			##################################################################################################	
			lfp = pd.Series(index = timestep, data = fp[:,ch])
			
			# TO REMOVE
			lfp = lfp.loc[half_sleep.start[0]:half_sleep.end[0]]
			#
			
			##################################################################################################
			# FILTERING
			##################################################################################################
			signal			= butter_bandpass_filter(lfp, low_cut, high_cut, frequency, 6)
			squared_signal = np.square(signal)
			window = np.ones(windowLength)/windowLength

			nSS[:,1] += scipy.signal.filtfilt(window, 1, squared_signal)
			# nSS[:,1] += scipy.ndimage.filters.gaussian_filter1d(squared_signal, 30)
			SS[:,1] += squared_signal

			del lfp
			del signal
			del squared_signal

	nSS[:,1] /= len(shank_to_channel[2])*3
	SS[:,1] /= len(shank_to_channel[2])*3

	nSS = pd.DataFrame(index = dummy.index, data = nSS)
	SS = pd.DataFrame(index = dummy.index, data = SS)

	nSS2 = (nSS[0] - nSS[1])/(nSS[1]+1)
	SS2 = (SS[0] - SS[1])/(SS[1]+1)

	nSS3 = nSS2.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)	

	nSS3 = (nSS3 - np.mean(nSS3))/(np.std(nSS3))

	# startex = int(2447.466*1e6)
	# stopex  = int(2448.261*1e6)




	# figure()
	# subplot(211)
	# plot(SS2.loc[startex:stopex])
	# # [axvline(t) for t in [26083000, 26101000, 26121000]]
	# subplot(212)
	# # plot(nSS2.loc[startex:stopex])
	# plot(nSS3.loc[startex:stopex])
	# axhline(low_thresFactor)
	# show()

	# sys.exit()

	# # Removing point above 100000/
	# nSS = nSS[nSS<nSS_highcut]
	# nSS = (nSS - np.mean(nSS))/np.std(nSS)

	# figure()
	# ax = subplot(211)
	# plot(lfp.loc[startex:stopex])
	# subplot(212, sharex =  ax)
	# plot(nSS.loc[startex:stopex])
	# axhline(low_thresFactor)
	# show()

	
	######################################################l##################################
	# Round1 : Detecting Ripple Periods by thresholding normalized signal
	thresholded = np.where(nSS3 > low_thresFactor, 1,0)
	start = np.where(np.diff(thresholded) > 0)[0]
	stop = np.where(np.diff(thresholded) < 0)[0]
	if len(stop) == len(start)-1:
		start = start[0:-1]
	if len(stop)-1 == len(start):
		stop = stop[1:]


	################################################################################################
	# Round 2 : Excluding candidates ripples whose length < minRipLen and greater than Maximum Ripple Length
	if len(start):
		l = (nSS3.index.values[stop] - nSS3.index.values[start])/1000 # from us to ms
		idx = np.logical_and(l > minRipLen, l < maxRipLen)
	else:	
		print("Detection by threshold failed!")
		sys.exit()

	rip_ep = nts.IntervalSet(start = nSS3.index.values[start[idx]], end = nSS3.index.values[stop[idx]])
	# rip_ep = rip_ep.intersect(sws_ep)


	# figure()
	# plot(lfp.restrict(sleep_ep))
	# plot(lfp.restrict(rip_ep))
	# show()



	####################################################################################################################
	# Round 3 : Merging ripples if inter-ripple period is too short
	rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval/1000, time_units = 's')



	#####################################################################################################################
	# Round 4: Discard Ripples with a peak power < high_thresFactor and > limit_peak
	rip_max = []
	rip_tsd = []
	for s, e in rip_ep.values:
		tmp = nSS3.loc[s:e]
		rip_tsd.append(tmp.idxmax())
		rip_max.append(tmp.max())

	rip_max = np.array(rip_max)
	rip_tsd = np.array(rip_tsd)

	# tokeep = np.logical_and(rip_max > high_thresFactor, rip_max < limit_peak)

	# rip_ep = rip_ep[tokeep].reset_index(drop=True)
	# rip_tsd = nts.Tsd(t = rip_tsd[tokeep], d = rip_max[tokeep])

	rip_tsd = nts.Tsd(t = rip_tsd, d = rip_max)
	###########################################################################################################
	# Writing for neuroscope
	start = rip_ep.as_units('ms')['start'].values
	peaks = rip_tsd.as_units('ms').index.values
	ends = rip_ep.as_units('ms')['end'].values

	datatowrite = np.vstack((start,peaks,ends)).T.flatten()

	n = len(rip_ep)

	texttowrite = np.vstack(((np.repeat(np.array(['UFO start 1']), n)), 
							(np.repeat(np.array(['UFO peak 1']), n)),
							(np.repeat(np.array(['UFO stop 1']), n))
								)).T.flatten()

	#evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.rip'
	evt_file = os.path.join(path, name + '.evt.py.ufo')
	f = open(evt_file, 'w')
	for t, n in zip(datatowrite, texttowrite):
		f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
	f.close()	


	# SS2.to_hdf(path+'/Analysis/SS.h5', 'ss')