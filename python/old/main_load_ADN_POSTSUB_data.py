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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_ADN_POS.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



for s in datasets:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))

	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	
	# filename = os.path.join(data_directory, s, name+'.csv')
	# position = pd.read_csv(filename, index_col = 0, header = None, names = ['x', 'y', 'ry'])
	# position_file = os.path.join(path, 'Analysis', 'Position.h5')
	# store = pd.HDFStore(position_file, 'w')
	# store['position'] = position
	# store.close()

	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)	
	sleep_ep		= loadEpoch(path, 'sleep')	
	
	tuning_curves 		= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)	
	tuning_curves 		= smoothAngularTuningCurves(tuning_curves, 10, 2)
	tokeep, stat 		= findHDCells(tuning_curves)

	
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gold', 'indianred', 'dodgerblue']

	shank = shank.flatten()

	figure()
	count = 1
	for j in np.unique(shank):
		neurons = np.where(shank == j)[0]
		for k,i in enumerate(neurons):
			subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
			plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]])
			if i in tokeep:
				plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]], linewidth = 3)
			# legend()
			count+=1
			xticks()
				

	