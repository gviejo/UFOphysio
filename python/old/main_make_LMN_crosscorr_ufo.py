import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from pycircstat.descriptive import mean as circmean

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)

infoall = []
ccufos = []

# for s in datasets:
# for s in datasets:
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
	wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	sws_ep 			= loadEpoch(path, 'sws')
	rem_ep 			= loadEpoch(path, 'rem')
	ufo_ep, ufo_tsd	= loadUFOs(path)

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves = {1:computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)}
	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

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
	tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

	tcurves 							= tuning_curves[1][tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	neurons 							= [name+'_'+str(n) for n in spikes.keys()]

	info 								= pd.DataFrame(index = neurons, columns = ['shank', 'hd', 'peaks'], data = 0)
	info['shank'] 						= shank.flatten()
	info['peaks'].iloc[tokeep] 			= peaks.values
	info['hd'].iloc[tokeep]				= 1

	############################################################################################### 
	# CROSS CORRS
	############################################################################################### 	
	# auto, fr = compute_AutoCorrs({0:ufo_tsd}, sws_ep, binsize = 1, nbins = 2000)
	cc_ufo = compute_EventCrossCorr(spikes, ufo_tsd, sws_ep, binsize = 10, nbins = 200, norm=True)
	cc_ufo = cc_ufo.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1)


	for i, s in enumerate(np.unique(shank)):
		figure()
		for j, n in enumerate(np.where(shank==s)[0]):
			subplot(int(np.sqrt(np.sum(shank==s)))+1,int(np.sqrt(np.sum(shank==s)))+1,j+1)
			plot(cc_ufo.iloc[:,n])

	figure()
	for i, s in enumerate(np.unique(shank)):
		subplot(2,3,i+1)
		# plot(cc_ufo[np.array(list(spikes.keys()))[np.where(shank==s)[0]]], color = 'grey')
		me = cc_ufo[np.array(list(spikes.keys()))[np.where(shank==s)[0]]].mean(1)
		st = cc_ufo[np.array(list(spikes.keys()))[np.where(shank==s)[0]]].std(1)
		plot(me)
		fill_between(me.index.values, me-st, me+st, alpha = 0.5)
		title(s)

	show()

	############# 
	# TO SAVE
	#############
	cc_ufo.columns 						= neurons
	infoall.append(info)
	ccufos.append(cc_ufo)
	#############

	# continue
	sys.exit()

	ufo_tsd = ufo_tsd.restrict(sws_ep)

	t = np.array_split(ufo_tsd.index.values, len(ufo_tsd)/30)

	colors = np.ones((len(peaks), 3))
	colors[:,0] = peaks.values/(2*np.pi)

	# colors = np.array([hsluv.hsluv_to_rgb(colors[i]) for i in range(len(colors))])
	colors = hsv_to_rgb(colors)

	rang = 250000
	for i in range(len(t)):
		figure()
		for j , tt in enumerate(t[i][0:30]):
			subplot(5,6,j+1)
			for l, k in enumerate(peaks.index.values):
				plot(spikes[k].loc[tt-rang:tt+rang].fillna(peaks[k]), '|', ms = 7, mew = 1.5, color = colors[l])
			axvline(tt, alpha = 0.4)
			xlim(tt-rang,tt+rang)
			ylim(0, 2*np.pi)
		show()
		# sys.exit()


infoall = pd.concat(infoall)
ccufos = pd.concat(ccufos, 1)

ccufos = ccufos.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1)


from umap import UMAP
ump = UMAP(n_neighbors = 10).fit_transform(ccufos.values.T)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(ccufos.values.T)

from sklearn.cluster import KMeans
K = KMeans(n_clusters = 3, random_state = 0).fit(ump).labels_


figure()
scatter(ump[:,0], ump[:,1], 100, c = K)
scatter(ump[:,0], ump[:,1], 10, c = infoall['hd'])

figure()
for i in np.unique(K):
	subplot(1,len(np.unique(K)),i+1)
	plot(ccufos.iloc[:,K==i], color = 'grey', alpha = 0.6)
	plot(ccufos.iloc[:,K==i].mean(1))

show()

