# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 08:52:32 2022

@author: ING57
"""

import h5py
import numpy as np

def ish5dataset(item):
	return isinstance(item, h5py.Dataset)


def h5printR(item, leading = ''):
	for key in item:
		if ish5dataset(item[key]):
			print(leading + key + ': ' + str(item[key].shape))
		else:
			print(leading + key)
			h5printR(item[key], leading + '  ')

# Print structure of a .doric file            
def h5print(filename):
	with h5py.File(filename, 'r') as h:
		print(filename)
		h5printR(h, '  ')


def h5read(filename,where):
	data = []
	with h5py.File(filename, 'r') as h:
		item = h
		for w in where:
			if ish5dataset(item[w]):
				data = np.array(item[w])
				DataInfo = {atrib: item[w].attrs[atrib] for atrib in item[w].attrs}
			else:
				item = item[w]
	
	return data, DataInfo


def h5getDatasetR(item, leading = ''):
	r = []
	for key in item:
		# First have to check if the next layer is a dataset or not
		firstkey = list(item[key].keys())
		if len(firstkey) == 0:
			continue
		else:
			firstkey = firstkey[0]
		if ish5dataset(item[key][firstkey]):
			r = r+[{'Name':leading+'_'+key, 'Data':
												[{'Name': k, 'Data': np.array(item[key][k]),
													'DataInfo': {atrib: item[key][k].attrs[atrib] for atrib in item[key][k].attrs}} for k in item[key]]}]
		else:
			r = r+h5getDatasetR(item[key], leading + '_' + key)
	
	return r


# Extact Data from a doric file
def ExtractDataAcquisition(filename):
	with h5py.File(filename, 'r') as h:
		#print(filename)
		return h5getDatasetR(h['DataAcquisition'],filename)
