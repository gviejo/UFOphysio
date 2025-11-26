import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from matplotlib.pyplot import *
from xml.dom import minidom
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import math
import nwbmatic as ntm
from sklearn.cluster import KMeans

# NOTES: 
    # 1) FOR CONTROL AND DEEPEST_DENTATE: MAKE SURE TO PICK CHANNELS THAT ARE NOT NOISY OR DEAD.
    # 2) MAKE SURE TO EXCLUDE NOISY OR DEAD CHANNELS FROM THE XML SPIKE GROUPS ("REMOVE CHANNELS FROM GROUP")
    
data_directory = '/media/guillaume/Elements1/B5500/B5501-251006B'
max_dentate_channel = 34 #pick channel where dentate spike is most visible in this recording
deepest_dentate = 58 #pick the channel where the dentate ends, otherwise the bottom-most channel
max_theta =6 #channel above the pyramidal layer with the cleanest theta
control = 59 #channel IN THE DENTATE, that does not have a visible dentate spike. Ideally a few channels below where ds2 changes polarity
std_threshold = 4.5 #dentate spikes need to go above this threshold to be detected

#Load xml file
xmlpath = os.path.join(data_directory, os.path.split(data_directory)[-1] + '.xml')
xmldoc = minidom.parse(xmlpath)

######################
# Load data
data = ntm.load_session(os.path.join(data_directory), 'neurosuite')

# sws = data.read_neuroscope_intervals(name = 'sws', path2file = os.path.join(data_directory, os.path.split(data_directory)[-1] + '.sws.evt'))
# rem = data.read_neuroscope_intervals(name = 'rem', path2file = os.path.join(data_directory, os.path.split(data_directory)[-1] + '.rem.evt'))
# sleep_ep =  sws.union(rem)#optional: restrict to when animal was sleeping


# Load EEG and clean up
frequency = float(xmldoc.getElementsByTagName("fieldPotentials")[0].getElementsByTagName("lfpSamplingRate")[0].firstChild.data)
lfp_filtered = nap.load_eeg(os.path.join(data_directory, os.path.split(data_directory)[-1] + '.eeg'),
                   n_channels = 64,frequency = frequency) #load
lfp_filtered = nap.apply_bandpass_filter(lfp_filtered, (5,100)) #bandpass filter

########################################################################################################
#FOR B4801 - CHANNEL IN DENTATE WHERE POLARITY SHIFTS IS VERY NOISY, FIX
########################################################################################################
# lfp_filtered[:,42] = (lfp_filtered[:,53].d + lfp_filtered[:,57]).d / 2


print('LFP loaded and bandpass filtered')

# Detect denate spikes
corrected_lfp = nap.Tsd(t = lfp_filtered.t, d = lfp_filtered[:, max_dentate_channel].d - lfp_filtered[:,control].d)
std_dentate = np.std(corrected_lfp) #get the std for the channel with the best dentate spikes
peaks = corrected_lfp.threshold(np.mean(corrected_lfp) + (std_dentate * std_threshold), 'above') #find peaks that exceed threshold, actual paper used 4.0-4.5std
peaks = nap.Ts([peaks.restrict(i).t[np.argmax(peaks.restrict(i))] for i in peaks.time_support])
print(peaks[0:10], peaks.shape) #check the first 10 dentate spikes

######################
#Characterise Dentate spikes
xml_channels = xmldoc.getElementsByTagName('spikeDetection')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')[0].getElementsByTagName('channels')[0].getElementsByTagName('channel')
xml_channels = np.array([int(channel.firstChild.data) for channel in xml_channels]) #channels ordered according to neuroscope, minus removed channels
xml_channels = xml_channels[np.argwhere(xml_channels == max_theta)[0][0] : np.argwhere(xml_channels == deepest_dentate)[0][0]+1]

#get the CSD of the array of channels from max_theta to deepest_dentate, in timespan of dentate spike +- 100 ms
######################
#We only need to compute csd for our dentate spikes
ds_csd = []
mean_lfp = []
vector_csd = []
included_ds = [] #keep track of ds that were excluded
ds_window = 0.100 #seconds on each side of dentate spike
for ds in peaks.t:
    ds_interval = nap.IntervalSet(start = [ds - ds_window], end = [ds + ds_window]) #get lfp +- 100 ms
    ds_lfp = lfp_filtered.restrict(ds_interval).loc[xml_channels] #restrict filtered lfp and take only relevant channels
    
    #CSD
    # f_matrix = np.eye(ds_lfp.d.T.shape[0])
    # csd = np.linalg.solve(f_matrix, ds_lfp.d.T)
    
    #dupret paper csd
    csd = np.array([(2*ds_lfp[:,channel].d) - ds_lfp[:,channel+1].d - ds_lfp[:,channel-1].d for channel in np.arange(1,len(ds_lfp.columns)-1)])
    csd = (csd - csd.min()) / (csd.max() - csd.min()) #normalise
    csd = (csd - csd.mean()) / csd.std() #z-score

    #try z-scoring
    vcsd = csd[:,math.floor(csd.shape[1]/2)]
    vector_csd.append(vcsd)
    
    if ds_lfp.shape[0] != int(ds_window*2*frequency+1): #skip this dentate spike for group figs if the whole window is not present
        included_ds.append(False)
    else:
        included_ds.append(True)
    
    mean_lfp.append(ds_lfp)
    ds_csd.append(csd.astype(float))
    
    continue
    figure() #figure just to check if this makes sense
    imshow(csd[::-1,:], cmap = 'PiYG', aspect = 'auto', origin = 'lower')
    for x, ch in enumerate(ds_lfp.columns):
        if ch == max_dentate_channel:
            plot((ds_lfp.loc[ch].d/1000) + ((len(ds_lfp.columns) - x-2)), c = 'black', linewidth = 3, alpha = 0.5)
        plot((ds_lfp.loc[ch].d/1000) + ((len(ds_lfp.columns) - x-2)), c = 'black', alpha = 0.5)
    colorbar(ticks = [np.min(csd), np.max(csd)], format = mticker.FixedFormatter(['Sink', 'Source']))
    tight_layout()
    show()
    sys.exit()

print('Dentate spike detection done, categorizing...')

if not os.path.exists(os.path.join(data_directory, 'DS_detection')): #check if fig folder exists already, make it if not
    os.makedirs(os.path.join(data_directory, 'DS_detection'))

figure()
title('Average DS')
mean_ds = np.mean([n for i,n in enumerate(mean_lfp) if included_ds[i] == True], axis = 0)
mean_csd = np.mean([n for i,n in enumerate(ds_csd) if included_ds[i] == True], axis = 0)
imshow(mean_csd[::-1,:], cmap = 'PiYG', aspect = 'auto', origin = 'lower', vmin = -max(mean_csd.max(), mean_csd.min()), vmax = max(mean_csd.max(), mean_csd.min()))
for x, ch in enumerate(mean_ds.T):
    if x == np.where(xml_channels == max_dentate_channel)[0][0]:
        plot((ch/1000) + (mean_ds.shape[1] - x), c = 'black', alpha = 0.5, linewidth = 3)
    plot((ch/1000) + (mean_ds.shape[1] - x), c = 'black', alpha = 0.5)
colorbar(ticks = [np.min(mean_csd), np.max(mean_csd)], format = mticker.FixedFormatter(['Sink', 'Source']))
tight_layout()
savefig(os.path.join(data_directory, 'DS_detection','average_ds.pdf'), format = 'pdf')
show()


######################
#PCA
#input must be (n_samples, n_features)
pca = PCA()
# pca.fit([i.d.flatten() for i in ds_csd])
ds_pca = pca.fit_transform(np.array(vector_csd)) #take only the peak time for pca

######################
#classify first two components into groups
# gm = GaussianMixture(n_components = 2)
# gm = DBSCAN(eps = 1, min_samples = 38)
# ds_classified = gm.fit_predict(ds_pca[:,[0,1]])

gm = KMeans(n_clusters = 2).fit(ds_pca[:,[0,1]])
ds_classified = gm.labels_
colortest = {0:'red', 1: 'blue', -1: 'green'}
colortest = [colortest[x] for x in ds_classified]

figure()
title('PCA on CSD')
scatter(ds_pca[:,0], ds_pca[:,1], c = colortest)
savefig(os.path.join(data_directory, 'DS_detection','csd_pca.pdf'), format = 'pdf')
show()


######################
#Divide PCA into groups
ds_class_no_ex = np.array([n for i,n in enumerate(ds_classified) if included_ds[i] == True])

ds1 = np.mean(np.array([n for i,n in enumerate(ds_csd) if included_ds[i] == True])[ds_class_no_ex == 0], axis = 0)
lfp1 = np.mean(np.array([n for i,n in enumerate(mean_lfp) if included_ds[i] == True])[ds_class_no_ex == 0], axis = 0)
centerds1 = ds1[:,math.floor(lfp1.shape[0]/2)]

ds2 = np.mean(np.array([n for i,n in enumerate(ds_csd) if included_ds[i] == True])[ds_class_no_ex == 1], axis = 0)
lfp2 = np.mean(np.array([n for i,n in enumerate(mean_lfp) if included_ds[i] == True])[ds_class_no_ex == 1], axis = 0)
centerds2 = ds2[:,math.floor(lfp2.shape[0]/2)]

#see which group has a higher sink
sink_loc = [np.argmin(centerds1), np.argmin(centerds2)]
if np.argmin(sink_loc) == 0: #if group ds1 has a higher sink than ds2
    ds_groups = {0: 'DS1', 1: 'DS2'}
    ds_class = {0: 'DS1', 1: 'DS2', -1: 'Unclassified'}
    ds_labels = [ds_class[i] for i in ds_classified]
    
elif np.argmin(sink_loc) == 1: #if group ds2 has a higher sink than ds1
    ds_groups = {0: 'DS2', 1: 'DS1'}
    ds_class = {0: 'DS2', 1: 'DS1', -1: 'Unclassified'}
    ds_labels = [ds_class[i] for i in ds_classified]

######################
#CHECK CLASSIFIER
figure()
subplot(1,2,1)
title(ds_groups[0] + ', red points')
imshow(ds1[::-1,:], cmap = 'PiYG', aspect = 'auto', origin = 'lower', vmin = -max(ds1.max(), ds1.min()), vmax = max(ds1.max(), ds1.min()))
for x, ch in enumerate(lfp1.T):
    plot((ch/1000) + (lfp1.shape[1] - x - 2), c = 'black', alpha = 0.5)
colorbar(ticks = [np.min(ds1), np.max(ds1)], format = mticker.FixedFormatter(['Sink', 'Source']))
scatter(5,np.argmin(centerds1[::-1]), marker = 5, s = 3000, c = 'grey')
subplot(1,2,2)
title(ds_groups[1] + ', blue points')
imshow(ds2[::-1,:], cmap = 'PiYG', aspect = 'auto', origin = 'lower', vmin = -max(ds2.max(), ds2.min()), vmax = max(ds2.max(), ds2.min()))
for x, ch in enumerate(lfp2.T):
    plot((ch/1000) + (lfp2.shape[1] - x - 2), c = 'black', alpha = 0.5)
colorbar(ticks = [np.min(ds2), np.max(ds2)], format = mticker.FixedFormatter(['Sink', 'Source']))
scatter(5,np.argmin(centerds2[::-1]), marker = 5, s = 3000, c = 'grey')
tight_layout()
savefig(os.path.join(data_directory, 'DS_detection','ds_classification.pdf'), format = 'pdf')
show()


######################
#Write csv with dentate spike times
pd.DataFrame(data = [peaks.t, ds_labels], index = ['DS time (s)', 'Type']).T.to_csv(os.path.join(data_directory,'dentate_spike_times.csv'))
# peaks.as_series().to_csv(os.path.join(data_directory,'dentate_spike_times.csv'), header = 'DS time (s)')

#Write parameters used for detection
pd.DataFrame(data = [max_dentate_channel, deepest_dentate, max_theta, control, std_threshold], 
             index = ['max_dentate_channel', 'deepest_dentate', 'max_theta', 'control', 'std_threshold']).T.to_csv(os.path.join(data_directory,'DS_detection', 'variables_used.csv'))


#write for neuroscope
ds_nsc = peaks.as_units('ms').index.values
texttowrite = ds_labels

evt_file = os.path.join(data_directory, os.path.split(data_directory)[-1]+ '.DST.evt')
f = open(evt_file, 'w')
for t, n in zip(ds_nsc, texttowrite):
    f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
f.close()   


# ds_lfp = nap.TsdFrame(data = [])
# csd_lfp = nap.apply_bandpass_filter(lfp[:,xml_channels])
# f_matrix = np.eye()