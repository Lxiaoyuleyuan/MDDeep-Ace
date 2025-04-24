# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:30:04 2022

@author: Windows User
"""

"""
this code reads data in CSV format, saves the labels in the data as a NumPy file, 
resizes the sequential features in the data to a size of 31*21, and saves them as a NumPy file.
"""
import numpy as np
import os



filefolder = 'data_ace/data/' #cvs file path

window_size = 41
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

datafolder = 'data_ace/data_npy/'#numpy file path
if not os.path.exists(datafolder):
        os.makedirs(datafolder)
        
def get_data(filefolder,species,phase,window_size):
    datafile = '{:s}-{:s}-{:d}_code.csv'.format(phase,species,window_size)
    filepath = os.path.join(filefolder, datafile)
    data = np.loadtxt(filepath, delimiter=',', dtype=np.int32)
    label = data[:, 0]
    label = np.expand_dims(label, axis=1)
    labels = convert_to_one_hot(label,2)
    labels = np.transpose(labels, axes=[1, 0])
    features = data[:, 1:]
    features = np.array([r.reshape((window_size, 21)) for r in features], dtype=np.float32)
    features = np.transpose(features,(0,2,1))
    if phase == 'val':
        phase = 'valid'
    np.save(datafolder + '{:s}-{:s}-{:d}_feature.npy'.format(phase,species,window_size), features)
    np.save(datafolder + '{:s}-{:s}-{:d}_label.npy'.format(phase,species,window_size), labels)
    return features, labels

def get_data_unlabel(filefolder,species,phase,window_size):
    datafile = '{:s}-{:s}-{:d}_code.csv'.format(phase,species,window_size)
    filepath = os.path.join(filefolder, datafile)
    data = np.loadtxt(filepath, delimiter=',', dtype=np.int32)
    labels = data[:, 0]
    features = data[:, 1:]
    features = np.array([r.reshape((window_size, 21)) for r in features], dtype=np.float32)
    features = np.transpose(features,(0,2,1))
    if phase == 'val':
        phase = 'valid'
    np.save(datafolder + '{:s}-{:s}-{:d}_feature.npy'.format(phase,species,window_size), features)
    np.save(datafolder + '{:s}-{:s}-{:d}_label.npy'.format(phase,species,window_size), labels)
    return features, labels
# species name    
species_all = ['Homo_sapiens','Rattus_norvegicus','Schistosoma_japonicum','Saccharomyces_cerevisiae','Mus_musculus','Escherichia_coli','Bacillus_velezensis','Plasmodium_falciparum','Oryza_sativa','Arabidopsis_thaliana']
#train valid and test 
phase_list = ['test','valid','train']

for i in range(len(species_all)): #len(species_all)
    for j in range(len(phase_list)):
        features_t, labels_t = get_data(filefolder,species_all[i],phase = phase_list[j],window_size = window_size)
        print(features_t.shape)
      


