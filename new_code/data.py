from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

import sys
import argparse
import os
import glob 
import dcor

def get_data(filename):
    ## Load the data

    file_idx = np.genfromtxt(filename, dtype='str') 
    # fold_idx = np.loadtxt('fold.txt') #### Not necessary if we don't use CV
    dx = np.loadtxt('dx.txt')
    np.random.seed(seed=0)

    subject_num = file_idx.shape[0]
    print(subject_num)

    patch_x = 64
    patch_y = 64
    patch_z = 64
    min_x = 0 
    min_y = 0 
    min_z = 0

    augment_size = 1024
    augment_size_val = 256
    augment_size_test = 256
    data = np.zeros((subject_num, patch_x, patch_y, patch_z,1))
    i = 0

    for subject_idx in file_idx:
        filename_full = '/fs/neurosci01/qingyuz/3dcnn/ADNI/img_64_longitudinal/' + subject_idx

        img = nib.load(filename_full)
        img_data = img.get_fdata()

        data[i,:,:,:,0] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data[i,:,:,:,0] = (data[i,:,:,:,0] - np.mean(data[i,:,:,:,0])) / np.std(data[i,:,:,:,0])
        i += 1

    train_data = data[:900]
    train_dx = dx[:900]

    test_data = data[900:1100]
    test_dx = dx[900:1100]

    test_data = data[1100:]
    test_dx = dx[1100:]

    # Augment data
    train_data_pos = train_data[train_dx==1];
    train_data_neg = train_data[train_dx==0];

    train_data_pos_aug = augment_by_transformation(train_data_pos, augment_size)
    train_data_neg_aug = augment_by_transformation(train_data_neg, augment_size)

    train_data_aug = np.concatenate((train_data_neg_aug, train_data_pos_aug), axis=0)
    train_dx_aug = np.zeros((augment_size * 2,))
    train_dx_aug[augment_size:] = 1

    val_data_pos = val_data[train_dx==1];
    val_data_neg = val_data[train_dx==0];

    val_data_pos_aug = augment_by_transformation(val_data_pos, augment_size_val)
    val_data_neg_aug = augment_by_transformation(val_data_neg, augment_size_val)

    val_data_aug = np.concatenate((val_data_neg_aug, val_data_pos_aug), axis=0)
    val_dx_aug = np.zeros((augment_size_val * 2,))
    val_dx_aug[augment_size:] = 1

    test_data_pos = test_data[test_dx==1];
    test_data_neg = test_data[test_dx==0];

    test_data_pos_aug = augment_by_transformation(test_data_pos, augment_size_test)
    test_data_neg_aug = augment_by_transformation(test_data_neg, augment_size_test)

    test_data_aug = np.concatenate((test_data_neg_aug, test_data_pos_aug), axis=0)
    test_dx_aug = np.zeros((augment_size_test * 2,))
    test_dx_aug[augment_size_test:] = 1

    return train_data, val_data, test_data

## Data Augmentation
def augment_by_transformation(data,n):
    augment_scale = 1

    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]
        m = n - raw_n
        new_data = np.zeros((m,data.shape[1],data.shape[2],data.shape[3],1))
        for i in range(0,m):
            idx = np.random.randint(0,raw_n)
            new_data[i] = data[idx].copy()
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(1,0),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(0,2),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(1,2),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.shift(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5))

        data = np.concatenate((data, new_data), axis=0)
        return data

## print(get_data('subjects_idx.txt')) #### In case you want to check if the get_data function works properly