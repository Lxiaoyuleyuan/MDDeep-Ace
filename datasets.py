# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:22:37 2022

@author: Windows User
"""

import torch
import torch.utils.data as data
import os
import numpy as np
import shutil
import random
import fnmatch
import glob
from torchvision import transforms as T
from numpy import random as nr

""" param """



def sample_data(data_path, species, mode):
    path_feature = data_path + mode + '-' + species + '-31_feature.npy' 
    path_label = data_path + mode + '-' + species + '-31_label.npy' 

    feature_all = np.load(path_feature)
    label_all = np.load(path_label) 
    n = label_all.shape[0]
    X=torch.Tensor(n,1,31,21)
    Y=torch.Tensor(n,2)

    inds=torch.randperm(n)
    for i,index in enumerate(inds):
        x=feature_all[index]
        y=label_all[index]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        X[i]=x
        Y[i]=y
    return X,Y



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
            self.path_feature = self.next_input[0]
            self.path_label = self.next_input[1]

        except StopIteration:
            self.next_input = None
            self.path_feature = None
            self.path_label = None
            return

            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input 
        feature = input[0]
        label = input[1]
        self.preload()
        return feature,label

    
    
class dataset_stage1(data.Dataset):
    def __init__(self,data_path, species1,species2, mode = 'train'):
        self.mode = mode
        self.data_path = data_path
        self.species1 = species1
        self.species2 = species2
        
        if self.mode == 'train':
            path_feature1 = self.data_path + self.mode + '-' + self.species1 + '-31_feature.npy'            
            self.path_feature1 = path_feature1
            path_label1 = self.data_path + self.mode + '-' + self.species1 + '-31_label.npy'            
            self.path_label1 = path_label1

            
            self.features1, self.labels1 = self.augmentation(rotation = True, flipping = True)
            
        if self.mode == 'test':
            path_feature1 = self.data_path + self.mode + '-' + self.species2 + '-31_feature.npy'            
            self.path_feature1 = path_feature1
            path_label1 = self.data_path + self.mode + '-' + self.species2 + '-31_label.npy'            
            self.path_label1 = path_label1

            
            self.features1, self.labels1 = self.augmentation(rotation = True, flipping = True)
             
        if self.mode == 'valid':
            path_feature1 = self.data_path + self.mode + '-' + self.species2 + '-31_feature.npy'            
            self.path_feature1 = path_feature1
            path_label1 = self.data_path + self.mode + '-' + self.species2 + '-31_label.npy'            
            self.path_label1 = path_label1
            
            self.features1, self.labels1 = self.augmentation(rotation = True, flipping = True)
            
            
    def augmentation(self, rotation = False, flipping = False):
        features1, labels1 = [], []
        
        feature_path1 = self.path_feature1        
        label_path1 = self.path_label1      
        feature_all1 = np.load(feature_path1)       
        label_all1 = np.load(label_path1) 
 
        for i in range(label_all1.shape[0]):
            features1.append(feature_all1[i])
            labels1.append(label_all1[i])
      
        return features1, labels1
    
    def __getitem__(self, index):
 
        features1 = self.features1
        labels1 = self.labels1
        
        feature_singel1 = features1[index]
        label_singel1 = labels1[index]

        
        return feature_singel1, label_singel1
    
    def __len__(self):
        return len(self.features1)
        

            
            
            
class dataset_stage2(data.Dataset):
    def __init__(self,data_path, species_list,species_target, mode = 'train'):
        self.mode = mode
        self.data_path = data_path
        self.species_list = species_list
        self.species_target = species_target
        if self.mode == 'train':

            path_feature_target = self.data_path + self.mode + '-' + self.species_target + '-31_feature.npy'            
            self.path_feature_target = path_feature_target
            path_label_target = self.data_path + self.mode + '-' + self.species_target + '-31_label.npy'            
            self.path_label_target = path_label_target

            path_feature1 = self.data_path + self.mode + '-' + self.species_list[1] + '-31_feature.npy'            
            self.path_feature1 = path_feature1
            path_label1 = self.data_path + self.mode + '-' + self.species_list[1] + '-31_label.npy'            
            self.path_label1 = path_label1
            
            path_feature0 = self.data_path + self.mode + '-' + self.species_list[0] + '-31_feature.npy'            
            self.path_feature0 = path_feature0
            path_label0 = self.data_path + self.mode + '-' + self.species_list[0] + '-31_label.npy'            
            self.path_label0 = path_label0

            path_feature2 = self.data_path + self.mode + '-' + self.species_list[2] + '-31_feature.npy'            
            self.path_feature2 = path_feature2
            path_label2 = self.data_path + self.mode + '-' + self.species_list[2] + '-31_label.npy'            
            self.path_label2 = path_label2

            path_feature3 = self.data_path + self.mode + '-' + self.species_list[3] + '-31_feature.npy'            
            self.path_feature3 = path_feature3
            path_label3 = self.data_path + self.mode + '-' + self.species_list[3] + '-31_label.npy'            
            self.path_label3 = path_label3

            path_feature4 = self.data_path + self.mode + '-' + self.species_list[4] + '-31_feature.npy'            
            self.path_feature4 = path_feature4
            path_label4 = self.data_path + self.mode + '-' + self.species_list[4] + '-31_label.npy'            
            self.path_label4 = path_label4
            
            path_feature5 = self.data_path + self.mode + '-' + self.species_list[5] + '-31_feature.npy'            
            self.path_feature5 = path_feature5
            path_label5 = self.data_path + self.mode + '-' + self.species_list[5] + '-31_label.npy'            
            self.path_label5 = path_label5

            path_feature6 = self.data_path + self.mode + '-' + self.species_list[6] + '-31_feature.npy'            
            self.path_feature6 = path_feature6
            path_label6 = self.data_path + self.mode + '-' + self.species_list[6] + '-31_label.npy'            
            self.path_label6 = path_label6

            path_feature7 = self.data_path + self.mode + '-' + self.species_list[7] + '-31_feature.npy'            
            self.path_feature7 = path_feature7
            path_label7 = self.data_path + self.mode + '-' + self.species_list[7] + '-31_label.npy'            
            self.path_label7 = path_label7

            path_feature8 = self.data_path + self.mode + '-' + self.species_list[8] + '-31_feature.npy'            
            self.path_feature8 = path_feature8
            path_label8 = self.data_path + self.mode + '-' + self.species_list[8] + '-31_label.npy'            
            self.path_label8 = path_label8
            
            self.features_target, self.labels_target, self.features0, self.labels0,self.features1, self.labels1,self.features2, self.labels2, self.features3, self.labels3,self.features4, self.labels4,self.features5, self.labels5,self.features6, self.labels6,self.features7, self.labels7,self.features8, self.labels8 = self.augmentation(rotation = True, flipping = True)
            

    def augmentation(self, rotation = False, flipping = False):
        features_target, labels_target = [], []
        features1, labels1 = [], []
        features2, labels2 = [], []
        features3, labels3 = [], []
        features4, labels4 = [], []
        features5, labels5 = [], []
        features6, labels6 = [], []
        features7, labels7 = [], []
        features8, labels8 = [], []
        features0, labels0 = [], []

        feature_path_target = self.path_feature_target     
        label_path_target = self.path_label_target     
        feature_all_target = np.load(feature_path_target)       
        label_all_target = np.load(label_path_target) 

        feature_path0 = self.path_feature0     
        label_path0 = self.path_label0      
        feature_all0 = np.load(feature_path0)       
        label_all0 = np.load(label_path0) 
        
        feature_path1 = self.path_feature1        
        label_path1 = self.path_label1      
        feature_all1 = np.load(feature_path1)       
        label_all1 = np.load(label_path1) 
        
        feature_path2 = self.path_feature2        
        label_path2 = self.path_label2      
        feature_all2 = np.load(feature_path2)       
        label_all2 = np.load(label_path2) 

        feature_path3 = self.path_feature3        
        label_path3 = self.path_label3      
        feature_all3 = np.load(feature_path3)       
        label_all3 = np.load(label_path3) 
        
        feature_path4 = self.path_feature4        
        label_path4 = self.path_label4      
        feature_all4 = np.load(feature_path4)       
        label_all4 = np.load(label_path4) 

        feature_path5 = self.path_feature5       
        label_path5 = self.path_label5     
        feature_all5 = np.load(feature_path5)       
        label_all5 = np.load(label_path5) 
        
        feature_path6 = self.path_feature6       
        label_path6 = self.path_label6     
        feature_all6 = np.load(feature_path6)       
        label_all6 = np.load(label_path6) 

        feature_path7 = self.path_feature7       
        label_path7 = self.path_label7    
        feature_all7 = np.load(feature_path7)       
        label_all7 = np.load(label_path7) 
        
        feature_path8 = self.path_feature8       
        label_path8 = self.path_label8    
        feature_all8 = np.load(feature_path8)       
        label_all8 = np.load(label_path8)
     
        for i in range(label_all_target.shape[0]):
            features_target.append(feature_all_target[i])
            labels_target.append(label_all_target[i])

        for i in range(label_all1.shape[0]):
            features1.append(feature_all1[i])
            labels1.append(label_all1[i])
            
        for i in range(label_all2.shape[0]):
            features2.append(feature_all2[i])
            labels2.append(label_all2[i])
            
        for i in range(label_all3.shape[0]):
            features3.append(feature_all3[i])
            labels3.append(label_all3[i])
            
        for i in range(label_all4.shape[0]):
            features4.append(feature_all4[i])
            labels4.append(label_all4[i])

        for i in range(label_all5.shape[0]):
            features5.append(feature_all5[i])
            labels5.append(label_all5[i])
            
        for i in range(label_all6.shape[0]):
            features6.append(feature_all6[i])
            labels6.append(label_all6[i])

        for i in range(label_all7.shape[0]):
            features7.append(feature_all7[i])
            labels7.append(label_all7[i])
            
        for i in range(label_all8.shape[0]):
            features8.append(feature_all8[i])
            labels8.append(label_all8[i])

            
        for i in range(label_all0.shape[0]):
            features0.append(feature_all0[i])
            labels0.append(label_all0[i])
            
        return features_target, labels_target, features0, labels0, features1, labels1, features2, labels2, features3, labels3, features4, labels4, features5, labels5, features6, labels6, features7, labels7, features8, labels8

    def __getitem__(self, index):
        features_target = self.features_target
        labels_target = self.labels_target
        features0 = self.features0
        labels0 = self.labels0
        features1 = self.features1
        labels1 = self.labels1
        features2 = self.features2
        labels2 = self.labels2
        features3 = self.features3
        labels3 = self.labels3
        features4 = self.features4
        labels4 = self.labels4
        features5 = self.features5
        labels5 = self.labels5
        features6 = self.features6
        labels6 = self.labels6
        features7 = self.features7
        labels7 = self.labels7
        features8 = self.features8
        labels8 = self.labels8
       
        feature_singel_target = features_target[index]
        label_singel_target = labels_target[index]

        l_t = len(labels_target)
        l = len(labels0)
        if l<=l_t:
            feature_singel0 = features0[index%l]
            label_singel0 = labels0[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel0 = features0[mylist[index]]
            label_singel0 = labels0[mylist[index]]
            
        l_t = len(labels_target)
        l = len(labels1)
        if l<=l_t:
            feature_singel1 = features1[index%l]
            label_singel1 = labels1[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel1 = features1[mylist[index]]
            label_singel1 = labels1[mylist[index]]

        l_t = len(labels_target)
        l = len(labels2)
        if l<=l_t:
            feature_singel2 = features2[index%l]
            label_singel2 = labels2[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel2 = features2[mylist[index]]
            label_singel2 = labels2[mylist[index]]

        l_t = len(labels_target)
        l = len(labels3)
        if l<=l_t:
            feature_singel3 = features3[index%l]
            label_singel3 = labels3[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel3 = features3[mylist[index]]
            label_singel3 = labels3[mylist[index]]

        l_t = len(labels_target)
        l = len(labels4)
        if l<=l_t:
            feature_singel4 = features4[index%l]
            label_singel4 = labels4[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel4 = features4[mylist[index]]
            label_singel4 = labels4[mylist[index]]

        l_t = len(labels_target)
        l = len(labels5)
        if l<=l_t:
            feature_singel5 = features5[index%l]
            label_singel5 = labels5[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel5 = features5[mylist[index]]
            label_singel5 = labels5[mylist[index]]

        l_t = len(labels_target)
        l = len(labels6)
        if l<=l_t:
            feature_singel6 = features6[index%l]
            label_singel6 = labels6[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel6 = features6[mylist[index]]
            label_singel6 = labels6[mylist[index]]

        l_t = len(labels_target)
        l = len(labels7)
        if l<=l_t:
            feature_singel7 = features7[index%l]
            label_singel7 = labels7[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel7 = features7[mylist[index]]
            label_singel7 = labels7[mylist[index]]

        l_t = len(labels_target)
        l = len(labels8)
        if l<=l_t:
            feature_singel8 = features8[index%l]
            label_singel8 = labels8[index%l]
        else:
            mylist = list(range(l))          
            random.shuffle(mylist)
            feature_singel8 = features8[mylist[index]]
            label_singel8 = labels8[mylist[index]]


        

        feature_singel = [feature_singel_target,feature_singel0,feature_singel1, feature_singel1,feature_singel2,feature_singel3,feature_singel4, feature_singel5,feature_singel6,feature_singel7,feature_singel8]
        label_singel = [label_singel_target,label_singel0,label_singel1,label_singel2,label_singel3,label_singel4,label_singel5,label_singel6,label_singel7,label_singel8]

        
        return feature_singel, label_singel
    
    def __len__(self):
        return len(self.features_target)        
    
 
