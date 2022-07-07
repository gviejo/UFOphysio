import numpy as np
import pandas as pd
from pylab import *
import sys, os
import h5py as hdf
from sklearn.decomposition import PCA

def loadXML(path):
	"""
	path should be the folder session containing the XML file
	Function returns :
		1. the number of channels
		2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
			eeg file first if both are present or both are absent
		3. the mappings shanks to channels as a dict
	Args:
		path : string

	Returns:
		int, int, dict
	"""
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	listdir = os.listdir(path)
	xmlfiles = [f for f in listdir if f.endswith('.xml')]
	if not len(xmlfiles):
		print("Folder contains no xml files; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, xmlfiles[0])
	
	from xml.dom import minidom	
	xmldoc 		= minidom.parse(new_path)
	nChannels 	= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
	fs_dat 		= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
	fs_eeg 		= xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data	
	if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
		fs = fs_dat
	elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
		fs = fs_eeg
	else:
		fs = fs_eeg
	shank_to_channel = {}
	groups 		= xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
	for i in range(len(groups)):
		shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
	return int(nChannels), int(fs), shank_to_channel




basepath = '/mnt/Data2/SpykingCircus/LMN/A5002/A5002-200303B_sh2'
basename = basepath.split('/')[-1]


n_channels, fs, shank_to_channel 	= loadXML(basepath)


# clusters = pd.HDFStore(os.path.join(basepath, basename, basename + '.clusters.hdf5'), 'r')
clusters = hdf.File(os.path.join(basepath, basename, basename + '.clusters.hdf5'), 'r')

clu = {}
res = {}

for i in shank_to_channel.keys():
	clu[i] = []
	res[i] = []
	count = 2
	for j in shank_to_channel[i]:
		tmp = clusters['clusters_'+str(j)][:]
		if len(tmp):			
			tmp = tmp - tmp.min()
			print(j,np.unique(tmp))
			n_clu = len(np.unique(tmp))
			tmp = tmp + count			
			clu[i].append(tmp)
			res[i].append(clusters['times_'+str(j)][:])

	clu[i] = np.hstack(clu[i])
	res[i] = np.hstack(res[i])
	clu[i] = np.hstack(([len(np.unique(clu[i]))],clu[i][np.argsort(res[i])]))
	res[i] = np.sort(res[i])


# Saving clu files
for i in clu.keys():
	np.savetxt(os.path.join(basepath, basename + '.clu.'+str(i+1)), clu[i], delimiter = '\n', fmt='%i')

# Saving res files
for i in res.keys():
	np.savetxt(os.path.join(basepath, basename + '.res.'+str(i+1)), res[i], delimiter = '\n', fmt='%i')	


# Saving spk files
spk = {}
f = open(os.path.join(basepath, basename + '.dat'), 'rb')
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
bytes_size = 2		
n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
duration = n_samples/20000
f.close()
fp = np.memmap(os.path.join(basepath, basename + '.dat'), np.int16, 'r', shape = (n_samples, n_channels))		
timestep = (np.arange(0, n_samples)/20000)*1e6
timestep = timestep.astype(np.int64)
for i in clu.keys():
	waveforms = np.zeros((len(clu[i])-1,30,len(shank_to_channel[i])), dtype=np.int16)
	for j in range(len(res[i])):
		idx = np.searchsorted(timestep, res[i][j])
		waveforms[j] = fp[idx-15:idx+15,shank_to_channel[i]]
	spk[i] = waveforms
	waveforms.flatten().tofile(os.path.join(basepath, basename + '.spk.'+str(i+1)))

# Saving fet files
for i in clu.keys():
	features = np.zeros((len(res[i]),8*3+3))
	waveforms = spk[i]
	for j in range(8):
		features[:,j*3:j*3+3] = PCA(n_components=3).fit_transform(waveforms[:,:,j])
	features = features.astype(np.int64)
	f = open(os.path.join(basepath, basename + '.fet.'+str(i+1)), 'w')
	f.writelines(str(features.shape[-1])+'\n')
	for j in range(len(features)):		
		f.writelines('\t'.join(features[j].astype('str'))+'\n')
	f.close()		

