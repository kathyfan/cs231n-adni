import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import pdb

# preprocess subject label and data
csv_path = '/fs/neurosci01/qingyuz/lab_data/demographics_lab.csv'
data_path = '/fs/neurosci01/qingyuz/lab_data/img_64_longitudinal/'

df_raw = pd.read_csv(csv_path, usecols=['subject', 'demo_diag'])

pdb.set_trace()
# load label, age, image paths
img_paths = glob.glob(data_path+'*.nii.gz')
img_paths = sorted(img_paths)
subj_data = {}
label_dict = {'C': 0, 'E': 1, 'H': 2, 'HE': 3}
for img_path in img_paths:
    subj_id = os.path.basename(img_path).split('-')[0]
    date = os.path.basename(img_path).split('-')[1].split('.')[0]
    date_struct = datetime.strptime(date, '%Y%m%d')
    rows = df_raw.loc[(df_raw['subject'] == subj_id)]
    if rows.shape[0] == 0:
        print('Missing label for', subj_id)
    else:
        # build dict
        label = rows.iloc[0]['demo_diag']
        if label not in label_dict.keys(): # ignore other labels
            continue
        if subj_id not in subj_data:
            subj_data[subj_id] = {'label': label_dict[label], 'date': [], 'date_start': date_struct, 'date_interval': [], 'img_paths': []}

        subj_data[subj_id]['date'].append(date)
        subj_data[subj_id]['date_interval'].append((date_struct - subj_data[subj_id]['date_start']).days / 365.)
        subj_data[subj_id]['img_paths'].append(img_path)

pdb.set_trace()
# count labels
max_timestep = 0
num_cls = [0,0,0,0]
num_ts = np.zeros((15,))
counts = np.zeros((4, 15))
for subj_id, info in subj_data.items():
    num_timestep = len(info['img_paths'])
    if len(info['date_interval']) != num_timestep:
        print('Different number of timepoint', subj_id)
    max_timestep = max(max_timestep, num_timestep)
    num_cls[info['label']] += 1
    num_ts[num_timestep] += 1
    counts[info['label'], num_timestep] += 1
print('Number of subjects: ', len(subj_data))
print('Max number of timesteps: ', max_timestep)
print('Number of each timestep', num_ts)
print('Number of each class', num_cls)
print(counts)

counts_cum = counts.copy()
for i in range(counts.shape[1]-2, 0, -1):
    counts_cum[:, i] += counts_cum[:, i+1]
print(counts_cum)

pdb.set_trace()

# augment data for save
def load_images(img_paths):
    imgs = np.zeros((len(img_paths), 64, 64, 64))   # (ts, 64, 64, 64)
    for idx, img_path in enumerate(img_paths):
        img_nib = nib.load(img_path)
        img = img_nib.get_fdata()
        imgs[idx,:,:,:] = (img - np.mean(img)) / np.std(img)
    return imgs

def augment_by_half(imgs):
    imgs_half = np.zeros((2, imgs.shape[0], 32, 64, 64))   # (2, ts, 32, 64, 64)
    for idx in range(imgs.shape[0]):
        img = imgs[idx, :, :, :]
        imgs_half[0, idx, :, :, :] = img[:32, :, :]
        imgs_half[1, idx, :, :, :] = np.flip(img[32:, :, :], 0) - np.zeros_like(img[32:, :, :])
    return imgs_half

def augment_by_flip(imgs):
    imgs_half = np.zeros((2, imgs.shape[0], 64, 64, 64))   # (2, ts, 64, 64, 64)
    for idx in range(imgs.shape[0]):
        img = imgs[idx, :, :, :]
        imgs_half[0, idx, :, :, :] = img
        imgs_half[1, idx, :, :, :] = np.flip(img, 0) - np.zeros_like(img)
    return imgs_half

def augment_images(imgs, num_augmentation=20):
    imgs_aug = np.zeros((num_augmentation, 2, imgs.shape[0], 64, 64, 64))
    for idx_aug in range(num_augmentation):
        # all timepoints apply same augmentation
        rotate_x = np.random.uniform(-2,2)
        rotate_y = np.random.uniform(-2,2)
        rotate_z = np.random.uniform(-2,2)
        shift = np.random.uniform(-2,2)

        for idx in range(imgs.shape[0]):
            img = imgs[idx, :, :, :]
            # rotate
            img = scipy.ndimage.interpolation.rotate(img, rotate_x, axes=(1,0), reshape=False)
            img = scipy.ndimage.interpolation.rotate(img, rotate_y, axes=(0,2), reshape=False)
            img = scipy.ndimage.interpolation.rotate(img, rotate_z, axes=(1,2), reshape=False)
            # shift
            img = scipy.ndimage.shift(img, shift)
            #flip
            imgs_aug[idx_aug, 0, idx, :, :, :] = img
            imgs_aug[idx_aug, 1, idx, :, :, :] = np.flip(img, 0) - np.zeros_like(img)

    imgs_aug = imgs_aug.reshape((num_augmentation*2, imgs.shape[0], 64, 64, 64))
    return imgs_aug


pdb.set_trace()
# save augmented/preprocessed images
h5_train = h5py.File('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/LAB/lab_augmented_new.h5')
h5_test = h5py.File('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/LAB/lab_original_new.h5')
num_augmentation = 20
num_subj = 0
subj_label = {}
subj_label_all = {}
subj_date_interval = {}
for subj_id, info in subj_data.items():
    num_subj += 1
    print(num_subj)
    imgs = load_images(info['img_paths'])
    #imgs_half = augment_by_half(imgs)
    imgs_flip = augment_by_flip(imgs)
    imgs_aug = augment_images(imgs, num_augmentation)
    #subj_data[subj_id]['imgs_test'] = imgs_half
    #subj_data[subj_id]['imgs_train'] = imgs_aug
    subj_label[subj_id] = info['label']
    subj_date_interval[subj_id] = info['date_interval']
    h5_train.create_dataset(subj_id, data=imgs_aug)
    h5_test.create_dataset(subj_id, data=imgs_flip)
np.save('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/LAB/lab_label.npy', subj_label)
np.save('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/LAB/lab_date_interval.npy', subj_date_interval)



pdb.set_trace()
