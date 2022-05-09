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

ufo_tsd	= ufo_tsd.restrict(sws_ep)

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
# CROSS CORRS
############################################################################################### 	
auto, fr = compute_AutoCorrs({0:ufo_tsd}, sws_ep, binsize = 20, nbins = 2000)

cc_ufo = compute_EventCrossCorr(spikes, ufo_tsd, sws_ep, binsize = 0.25, nbins = 600, norm=True)
cc_ufo = cc_ufo.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2)

baseline = cc_ufo.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30)

zcc_ufo = (cc_ufo - baseline) / np.sqrt(baseline.var())


figure()
for i, n in enumerate(tokeep):
	subplot(int(np.sqrt(len(tokeep)))+1,int(np.sqrt(len(tokeep)))+1,i+1)
	if n in adn:
		plot(cc_ufo[n], color = 'red')
		plot(baseline[n], '--', color = 'red')
	else:
		plot(cc_ufo[n], color = 'green')
		plot(baseline[n], '--', color = 'green')

figure()
for i, n in enumerate(tokeep):
	subplot(int(np.sqrt(len(tokeep)))+1,int(np.sqrt(len(tokeep)))+1,i+1)
	if n in adn:
		plot(zcc_ufo[n], color = 'red')		
	else:
		plot(zcc_ufo[n], color = 'green')		



figure()
subplot(2,3,1)
me2 = cc_ufo[lmn].mean(1).loc[-30:30]
st2 = cc_ufo[lmn].std(1).loc[-30:30]
plot(me2, color = 'green')
fill_between(me2.index.values, me2-st2, me2+st2, color = 'green', alpha = 0.2)
title('LMN')
ylabel('Raw')
subplot(2,3,2)
me1 = cc_ufo[adn].mean(1).loc[-30:30]
st1 = cc_ufo[adn].std(1).loc[-30:30]
plot(me1, color = 'red')
fill_between(me1.index.values, me1-st1, me1+st1, color = 'red', alpha = 0.2)
title('ADN')
subplot(2,3,3)
plot(me1, color = 'red')
fill_between(me1.index.values, me1-st1, me1+st1, color = 'red', alpha = 0.2)
plot(me2, color = 'green')
fill_between(me2.index.values, me2-st2, me2+st2, color = 'green', alpha = 0.2)

subplot(2,3,4)
me2 = zcc_ufo[lmn].mean(1).loc[-30:30]
st2 = zcc_ufo[lmn].std(1).loc[-30:30]
plot(me2, color = 'green')
fill_between(me2.index.values, me2-st2, me2+st2, color = 'green', alpha = 0.2)
title('LMN')
ylabel('Zscore')
subplot(2,3,5)
me1 = zcc_ufo[adn].mean(1).loc[-30:30]
st1 = zcc_ufo[adn].std(1).loc[-30:30]
plot(me1, color = 'red')
fill_between(me1.index.values, me1-st1, me1+st1, color = 'red', alpha = 0.2)
title('ADN')
subplot(2,3,6)
plot(me1, color = 'red')
fill_between(me1.index.values, me1-st1, me1+st1, color = 'red', alpha = 0.2)
plot(me2, color = 'green')
fill_between(me2.index.values, me2-st2, me2+st2, color = 'green', alpha = 0.2)


show()

