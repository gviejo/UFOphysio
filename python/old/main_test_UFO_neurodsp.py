import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from neurodsp.filt import filter_signal
from neurodsp.burst import detect_bursts_dual_threshold


############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)


for s in datasets[-1:]:
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

	############################################################
	# DAT FILE NAME
	############################################################

	if 'A5002' in name:
		datfile = '/mnt/DataGuillaume/LMN-ADN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
	elif 'A1407' in name:
		datfile = '/mnt/DataGuillaume/LMN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
		shank_to_channel = {i+2:shank_to_channel[i] for i in shank_to_channel.keys()}
	elif 'A5001' in name:
		datfile = '/mnt/DataGuillaume/LMN-ADN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
		shank_to_channel = {i-3:shank_to_channel[i] for i in [5, 6, 7, 8]}
	
	############################################################
	# DURATION DAT FILE
	############################################################
	frequency = 20000

	f = open(datfile, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2		
	n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
	duration = n_samples/frequency
	f.close()

	############################################################
	# DURATION DAT FILE
	############################################################
	fp = np.memmap(datfile, np.int16, 'r', shape = (n_samples, n_channels))
	timestep = (np.arange(0, n_samples)/frequency)*1e6
	timestep = timestep.astype(np.int)



	duree = len(timestep)	

	dummy = pd.Series(index = timestep, data = 0)

	# # #TO REMOVE
	half_sleep = sleep_ep.loc[[0]]
	duree = len(dummy.loc[half_sleep.start[0]:half_sleep.end[0]].index.values)
	dummy = dummy.loc[half_sleep.start[0]:half_sleep.end[0]]

	nSS = np.zeros((duree,2))
	SS = np.zeros((duree,2))

	for ch in shank_to_channel[2]:
		##################################################################################################
		# LOADING LFP
		##################################################################################################	
		lfp = pd.Series(index = timestep, data = fp[:,ch])
		# TO REMOVE
		lfp = lfp.loc[half_sleep.start[0]:half_sleep.end[0]]
		#
		sig_filt = filter_signal(lfp.values, frequency, 'bandpass', (800, 3000))

		bursting = detect_bursts_dual_threshold(lfp.values, frequency, (1,2), (800, 3000))

		plot_bursts(lfp.index.values, lfp.values, bursting)

		sys.exit()
