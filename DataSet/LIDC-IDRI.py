from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
#from PIL import Image

import os
import numpy as np
import gzip
import pickle

from collections import defaultdict

#from sklearn.decomposition import PCA


class LIDC(data.Dataset):
    def __init__(self, filepath=None,train=True):


        self.filepath=filepath

        self.train_data=np.load(filepath+r'\LIDC-IDRI_images_npy_mean_HU.npy')
        self.train_labels=np.load(filepath+r'\LIDC-IDRI_labels_train.npy')
        self.test_data=np.load(filepath+r'\LIDC-IDRI_images_npy_mean_HU.npy')
        self.test_labels=np.load(filepath+r'\LIDC-IDRI_labels_test.npy')

        self.train_labels=np.reshape(self.train_labels,self.train_labels.shape[0])
        self.test_labels=np.reshape(self.test_labels,self.test_labels.shape[0])

        if train==True:
            self.data=self.train_data
            self.labels=self.train_labels.tolist()
        else:
            self.data=self.test_data
            self.labels=self.test_labels.tolist()

        self.data=self.data.astype(np.float32)

        Index = defaultdict(list)
        for i, label in enumerate(self.labels):
            Index[label].append(i)

        self.Index = Index

        classes = list(set(self.labels))
        self.classes = classes



    def load(self,filepath):
        print('filepath',filepath)
        with open(filepath,'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]        

    def __getitem__(self, index):

        img, label = self.data[index], self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)






