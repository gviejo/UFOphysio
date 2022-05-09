import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.colors import hsv_to_rgb
# import hsluv
from pycircstat.descriptive import mean as circmean

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

s = 'LMN-ADN/A5011/A5011-201014A'



print(s)
name = s.split('/')[-1]
path = os.path.join(data_directory, s)
############################################################################################### 
# LOADING DATA
###############################################################################################
episodes 							= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
episodes[episodes != 'sleep'] 		= 'wake'
events								= list(np.where(episodes != 'sleep')[0].astype('str'))	
spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					
sws_ep								= loadEpoch(path, 'sws')
rem_ep								= loadEpoch(path, 'rem')

ufo_ep, ufo_tsd						= loadUFOs(path)

ufo_tsd = ufo_tsd.restrict(sws_ep)

# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

# CHECKING HALF EPOCHS
wake2_ep = splitWake(wake_ep)
tokeep2 = []
stats2 = []
tcurves2 = []
for i in range(2):
	# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
	tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
	tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
	tokeep, stat = findHDCells(tcurves_half)
	tokeep2.append(tokeep)
	stats2.append(stat)
	tcurves2.append(tcurves_half)

tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

# NEURONS FROM ADN	
if 'A5011' in s:
	adn = np.where(shank <=3)[0]
	lmn = np.where(shank ==5)[0]

adn 	= np.intersect1d(adn, tokeep)
lmn 	= np.intersect1d(lmn, tokeep)
tokeep 	= np.hstack((adn, lmn))
spikes 	= {n:spikes[n] for n in tokeep}

tcurves 		= tuning_curves[tokeep]
peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))


############################################################################################### 
# Bayesian Decoding
############################################################################################### 	
bin_size = 2 # ms
overlap = 0.5 # %
windows = 100 # ms
bins = np.arange(0, windows+bin_size, bin_size) - windows/2

w = scipy.signal.gaussian(31, 2)
rate = []

for i, t in enumerate(ufo_tsd.index.values):
	tbins = (t+bins*1000)
	tmp = []
	for n in tokeep:
		count, _ = np.histogram(spikes[n].index.values, tbins)
		count = np.convolve(count, w, mode = 'same')
		tmp.append(count)
	tmp = np.array(tmp).T	
	rate.append(tmp)

rate = np.vstack(rate)

for i, gr in enumerate([adn, lmn]):
	tcurves_array = tuning_curves[gr].values
	spike_counts_array = spike_counts.values
	proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

	part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
	if px is not None:
		part2 = px
	else:
		part2 = np.ones(tuning_curves.shape[0])
	#part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
	
	for i in range(len(proba_angle)):
		part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
		p = part1 * part2 * part3
		proba_angle[i] = p/p.sum() # Normalization process here

	proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
	proba_angle = proba_angle.astype('float')
	decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')



