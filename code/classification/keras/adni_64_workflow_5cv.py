from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, UpSampling3D, Input, ZeroPadding3D, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv3D, MaxPooling3D
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.constraints import unit_norm, max_norm
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam

import tensorflow as tf

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


# In[2]:
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

        # output an example
        #array_img = nib.Nifti1Image(np.squeeze(new_data[3,:,:,:,0]),np.diag([1, 1, 1, 1]))  
        #filename = 'augmented_example.nii.gz'
        #nib.save(array_img,filename)

        data = np.concatenate((data, new_data), axis=0)
        return data

class CLF():
    
    def __init__(self):

        optimizer = Adam(0.0002)

        L2_reg = 0.1
        ft_bank_baseline = 16
        latent_dim = 16

        # Build the feature encoder
        input_image = Input(shape=(64,64,64,1), name='input_image')
        feature = Conv3D(ft_bank_baseline, activation='relu', kernel_size=(3, 3, 3),padding='same')(input_image)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = Conv3D(ft_bank_baseline*2, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = Conv3D(ft_bank_baseline*4, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = Conv3D(ft_bank_baseline*2, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature_dense = Flatten()(feature)
        
        self.encoder = Model(input_image, feature_dense)

        # Build and Compile the classifer  
        #self.encoder.load_weights('encoder.h5');
        #self.encoder.trainable = False
        input_feature_clf = Input(shape=(2048,), name='input_feature_dense')
        feature_clf = Dropout(0.25)(input_feature_clf)
        feature_clf = Dense(latent_dim*4, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
        #feature_clf = Dropout(0.1)(feature_clf)
        feature_clf = Dense(latent_dim*2, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
        #feature_clf = Dropout(0.1)(feature_clf)
        prediction_score = Dense(1, name='prediction_score',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
        self.classifier = Model(input_feature_clf, prediction_score)

        # Build the entire workflow
        prediction_score_workflow = self.classifier(feature_dense)
        label_workflow = Activation('sigmoid', name='r_mean')(prediction_score_workflow)
        self.workflow = Model(input_image, label_workflow)
        self.workflow.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    def train(self, epochs, training, testing, testing_raw, batch_size=64, fold=0):
        [train_data_aug, train_dx_aug] = training
        [test_data_aug,  test_dx_aug ] = testing
        [test_data    ,  test_dx     ] = testing_raw

        test_data_aug_flip = np.flip(test_data_aug,1)
        test_data_flip = np.flip(test_data,1)

        idx_perm = np.random.permutation(int(train_data_aug.shape[0]/2))
        
        for epoch in range(epochs):

            # Select a random batch of images
            
            idx_perm = np.random.permutation(int(train_data_aug.shape[0]/2))
            idx = idx_perm[:int(batch_size/2)]
            idx = np.concatenate((idx,idx+int(train_data_aug.shape[0]/2)))

            training_feature_batch = train_data_aug[idx]
            dx_batch = train_dx_aug[idx]
            
            # ---------------------
            #  Train classifier
            # ---------------------

            c_loss = self.workflow.train_on_batch(training_feature_batch, dx_batch)

            # ---------------------
            #  flip & re-do everything
            # ---------------------

            training_feature_batch = np.flip(training_feature_batch,1)
            c_loss = self.workflow.train_on_batch(training_feature_batch, dx_batch)

            # Plot the progress
            if epoch % 100 == 0:
                c_loss_test_1 = self.workflow.evaluate(test_data_aug,      test_dx_aug, verbose = 0, batch_size = batch_size)    
                c_loss_test_2 = self.workflow.evaluate(test_data_aug_flip, test_dx_aug, verbose = 0, batch_size = batch_size)    

                # print results for balanced dataset
                print ("%d [Acc: %f,  Test Acc: %f Test Acc Flip: %f" % (epoch, c_loss[1], c_loss_test_1[1], c_loss_test_2[1], ))
                sys.stdout.flush()              
              

if __name__ == '__main__':
    ## load the data

    file_idx = np.genfromtxt('./subjects_idx.txt', dtype='str') 
    fold_idx = np.loadtxt('fold.txt')
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
    augment_size_test = 512

    data = np.zeros((subject_num, patch_x, patch_y, patch_z,1))
    i = 0
    for subject_idx in file_idx:
        filename_full = '/fs/neurosci01/qingyuz/3dcnn/ADNI/img_64_longitudinal/'+subject_idx

        img = nib.load(filename_full)
        img_data = img.get_fdata()

        data[i,:,:,:,0] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data[i,:,:,:,0] = (data[i,:,:,:,0] - np.mean(data[i,:,:,:,0])) / np.std(data[i,:,:,:,0])

        #array_img = nib.Nifti1Image(np.squeeze(data[i,:,:,:,0]),np.diag([1, 1, 1, 1]))  
        #filename = 'processed_example.nii.gz'
        #nib.save(array_img,filename)

        i += 1
    
    ## cross-validation

    for fold in range(5):
        train_idx = (fold_idx != fold)
        test_idx = (fold_idx == fold)
        print('training num ',np.sum(train_idx),' test num ',np.sum(test_idx))

        train_data = data[train_idx]
        train_dx = dx[train_idx]

        test_data = data[test_idx]
        test_dx = dx[test_idx]

        # augment data
        train_data_pos = train_data[train_dx==1];
        train_data_neg = train_data[train_dx==0];

        train_data_pos_aug = augment_by_transformation(train_data_pos,augment_size)
        train_data_neg_aug = augment_by_transformation(train_data_neg,augment_size)

        train_data_aug = np.concatenate((train_data_neg_aug, train_data_pos_aug), axis=0)
        train_dx_aug = np.zeros((augment_size * 2,))
        train_dx_aug[augment_size:] = 1

        test_data_pos = test_data[test_dx==1];
        test_data_neg = test_data[test_dx==0];

        test_data_pos_aug = augment_by_transformation(test_data_pos,augment_size_test)
        test_data_neg_aug = augment_by_transformation(test_data_neg,augment_size_test)

        test_data_aug = np.concatenate((test_data_neg_aug, test_data_pos_aug), axis=0)
        test_dx_aug = np.zeros((augment_size_test * 2,))
        test_dx_aug[augment_size_test:] = 1
 
        print("Begin Training fold")
        sys.stdout.flush()
        clf = CLF()
        clf.train(epochs=1501, training=[train_data_aug, train_dx_aug], testing=[test_data_aug, test_dx_aug], testing_raw=[test_data, test_dx], batch_size=64, fold=fold)

