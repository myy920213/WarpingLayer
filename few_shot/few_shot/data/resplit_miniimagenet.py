import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt

import h5py

_MINI_IMAGENET_DATASET_DIR = '/home/yma36/data/mini_imagenet'


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


file_train_categories_train_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_train.pickle')

data_train = load_data(file_train_categories_train_phase)
train_data = data_train['data']
print('data:', train_data.shape)
train_labels = data_train['labels']
print('labels:', train_labels[601:605])
file_val_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_val.pickle')
file_test_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_test.pickle')
data_val = load_data(file_val_categories_val_phase)
val_data = data_val['data']
val_labels = data_val['labels']
data_test = load_data(file_test_categories_test_phase)
test_data = data_test['data']
test_labels = data_test['labels']
all_data = np.concatenate([train_data, val_data, test_data], axis=0)
print('all_data_size:', all_data.shape)
all_labels = train_labels + val_labels + test_labels
print('all_label_size:', len(all_labels))
#print('list_labels:', set(all_labels))
all_labels = np.array(all_labels)
train_all_data = []
train_all_labels = []
test_all_data = []
test_all_labels = []
for i in range(100):
    #current_labels = (all_labels==i)
    loc = np.where( all_labels== i)[0]
    if i ==0:
        print('class_length:', len(loc))
        print('class:', loc)
    current_data = all_data[loc]
    current_labels = all_labels[loc]
    current_train_data = current_data[0:500]
    train_all_data.append(current_train_data)
    current_test_data = current_data[500:]
    test_all_data.append(current_test_data)
    current_train_labels = current_labels[0:500]
    train_all_labels.append(current_train_labels)
    current_test_labels = current_labels[500:]
    test_all_labels.append(current_test_labels)
    
train_all_data = np.concatenate(train_all_data, axis=0)
print('train_all_data_size:', train_all_data.shape)
test_all_data = np.concatenate(test_all_data, axis=0)
print('test_all_data_size:', test_all_data.shape)
train_all_labels = np.concatenate(train_all_labels)
print('train_all_labels_size:', train_all_labels.shape)
test_all_labels = np.concatenate(test_all_labels)
print('test_all_labels_size:', test_all_labels.shape)
full_data_train = {'data': train_all_data, 'labels': train_all_labels}
full_data_test = {'data': test_all_data, 'labels': test_all_labels}



'''
train_to_store = open("full_data_train.pickle", "wb")
pickle.dump(full_data_train, train_to_store)
train_to_store.close()

test_to_store = open("full_data_test.pickle", "wb")
pickle.dump(full_data_test, test_to_store)
test_to_store.close()
'''




    #train_set = loc[0:500]
    #test_set = loc[500:]
    #train_all_data.append(all_data)



'''
from PIL import Image
import numpy as np
data_img = train_data[601].astype(np.uint8)
img = Image.fromarray(data_img, 'RGB')
img.save('my.png')
img.show()
'''