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
csv_path = '/fs/neurosci01/qingyuz/3dcnn/ADNI/info/ADNI1AND2.csv'
csv_path_raw = '../data/ADNI/ADNIMERGE.csv'
data_path1 = '/fs/neurosci01/qingyuz/3dcnn/ADNI/img_64_longitudinal/'
#data_path2 = '/fs/neurosci01/qingyuz/3dcnn/ADNI/img_64_longitudinal_2/'
df = pd.read_csv(csv_path)
df_raw = pd.read_csv(csv_path_raw, usecols=['PTID', 'DX_bl', 'DX', 'EXAMDATE', 'AGE'])


'''
# check label and image match
pdb.set_trace()
subj_id_list = df.drop_duplicates('Subject_ID')['Subject_ID'].values.tolist()
img_paths_1 = [os.path.basename(img_path).split('-')[0] for img_path in sorted(glob.glob(data_path1+'*.nii.gz'))]
img_paths_2 = [os.path.basename(img_path).split('-')[0] for img_path in sorted(glob.glob(data_path2+'*.nii.gz'))]
img_paths = img_paths_1 + img_paths_2
for subj_id in subj_id_list:
    row = df.loc[(df['Subject_ID'] == subj_id)]
    num_age = row.drop_duplicates('Age').shape[0]
    num_img = img_paths.count(subj_id)
    #if num_age != num_img and num_img != 0:
    #    print(subj_id, num_age, num_img)
    num_label = row.drop_duplicates('DX_Group').shape[0]
    if num_label != 1:
        print(subj_id, row.drop_duplicates('DX_Group'))
'''

'''
# preprocessing label
subj_id_list = df_raw.Subject.unique()
pdb.set_trace()
for subj_id in subj_id_list:
    subj_id_row = df_raw.loc[df_raw['Subject']==subj_id]
    if subj_id_row.Group.unique().shape[0] != 1:
        print(subj_id)
'''

pdb.set_trace()
# load label, age, image paths
img_paths = glob.glob(data_path1+'*.nii.gz')
img_paths = sorted(img_paths)
subj_data = {}
label_dict = {'Normal': 0, 'NC': 0, 'CN': 0, 'MCI': 1, 'LMCI': 1, 'EMCI': 1, 'AD': 2, 'Dementia': 2, 'sMCI':3, 'pMCI':4}
nan_label_count = 0
nan_idx_list = []
for img_path in img_paths:
    subj_id = os.path.basename(img_path).split('-')[0]
    date = os.path.basename(img_path).split('-')[1] + '-' + os.path.basename(img_path).split('-')[2] + '-' + os.path.basename(img_path).split('-')[3].split('_')[0]
    date_struct = datetime.strptime(date, '%Y-%m-%d')
    rows = df_raw.loc[(df_raw['PTID'] == subj_id)]
    if rows.shape[0] == 0:
        print('Missing label for', subj_id)
    else:
        # matching date
        date_diff = []
        for i in range(rows.shape[0]):
            date_struct_now = datetime.strptime(rows.iloc[i]['EXAMDATE'], '%Y-%m-%d')
            date_diff.append(abs((date_struct_now - date_struct).days))
        i = np.argmin(date_diff)
        if date_diff[i] > 120:
            print('Missing label for', subj_id, date_diff[i], date_struct)
            continue
        
        # build dict
        if subj_id not in subj_data:
            subj_data[subj_id] = {'age': rows.iloc[i]['AGE'], 'label_all': [], 'label': label_dict[rows.iloc[i]['DX_bl']], 'date': [], 'date_start': date_struct, 'date_interval': [], 'img_paths': []}

        if rows.iloc[i]['EXAMDATE'] in subj_data[subj_id]['date']:
            print('Multiple image at same date', subj_id, rows.iloc[i]['EXAMDATE'])
            continue     

        subj_data[subj_id]['date'].append(rows.iloc[i]['EXAMDATE'])
        subj_data[subj_id]['date_interval'].append((date_struct - subj_data[subj_id]['date_start']).days / 365.)
        subj_data[subj_id]['img_paths'].append(img_path)
        if pd.isnull(rows.iloc[i]['DX']) == False:
            subj_data[subj_id]['label_all'].append(label_dict[rows.iloc[i]['DX']])
        else:
            nan_label_count += 1
            nan_idx_list.append([subj_id, len(subj_data[subj_id]['label_all'])])
            subj_data[subj_id]['label_all'].append(-1)

# fill nan
print(nan_label_count)
for subj in nan_idx_list:
    subj_data[subj[0]]['label_all'][subj[1]] = subj_data[subj[0]]['label_all'][subj[1]-1]
    if subj_data[subj[0]]['label_all'][subj[1]] == -1:
        print(subj)

# overall label, 0, 2, 3, 4
pMCI_count = 0
num_nc = 0
num_ad = 0
num_mci = 0
for subj_id in subj_data.keys():
    if len(list(set(subj_data[subj_id]['label_all']))) != 1:
        print(subj_id, subj_data[subj_id]['label_all'])
        if list(set(subj_data[subj_id]['label_all'])) == [1,2] or list(set(subj_data[subj_id]['label_all'])) == [2,1] or list(set(subj_data[subj_id]['label_all'])) == [0,1,2]:
            pMCI_count += 1
            subj_data[subj_id]['label'] = 4
        if list(set(subj_data[subj_id]['label_all'])) == [0,1] or list(set(subj_data[subj_id]['label_all'])) == [1,0]:
            subj_data[subj_id]['label'] = 3
        if list(set(subj_data[subj_id]['label_all'])) == [0,2] or list(set(subj_data[subj_id]['label_all'])) == [2,0]:
            subj_data[subj_id]['label'] = 2
    elif subj_data[subj_id]['label'] == 1:
        subj_data[subj_id]['label'] = 3
    label_all = np.array(subj_data[subj_id]['label_all'])
    num_nc += (label_all==0).sum()
    num_mci += (label_all==1).sum() 
    num_ad += (label_all==2).sum()        
print(pMCI_count)
print(num_nc, num_mci, num_ad)            
pdb.set_trace()


'''
# old
img_paths1 = glob.glob(data_path1+'*.nii.gz')
img_paths2 = glob.glob(data_path2+'*.nii.gz')
img_paths = sorted(img_paths1 + img_paths2)
subj_data = {}
label_dict = {'Normal': 0, 'MCI': 1, 'AD': 2}
for img_path in img_paths:
    subj_id = os.path.basename(img_path).split('-')[0]
    row = df.loc[(df['PTID'] == subj_id)]
    if row.shape[0] == 0:
        print('Missing label for', subj_id)
    else:
        row = row.drop_duplicates('Age')
        if subj_id not in subj_data:
            subj_data[subj_id] = {'age': row.iloc[0]['Age'], 'label': label_dict[row.iloc[0]['DX_Group']], 'date': [], 'img_paths': []}
        subj_data[subj_id]['date'].append(os.path.basename(img_path).split('-')[1] + '-' + os.path.basename(img_path).split('-')[2])
        subj_data[subj_id]['img_paths'].append(img_path)
'''

# max ts
#pdb.set_trace()
max_timestep = 0
num_cls = [0,0,0,0,0]
num_ts = [0,0,0,0,0,0,0,0,0]
counts = np.zeros((5, 8))
for subj_id, info in subj_data.items():
    num_timestep = len(info['img_paths'])
    if len(info['label_all']) != num_timestep or len(info['date_interval']) != num_timestep:
        print('Different number of timepoint', subj_id)
    max_timestep = max(max_timestep, num_timestep)
    num_cls[info['label']] += 1
    num_ts[num_timestep] += 1
    #if num_timestep == 6:
    #    print(subj_id, info['date'])
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
h5_train = h5py.File('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/adni_augmented_large.h5')
h5_test = h5py.File('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/adni_original_large.h5')
num_augmentation = 50
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
    subj_label_all[subj_id] = info['label_all']
    subj_date_interval[subj_id] = info['date_interval']
    h5_train.create_dataset(subj_id, data=imgs_aug)
    h5_test.create_dataset(subj_id, data=imgs_flip)
np.save('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/adni_label_large.npy', subj_label)
np.save('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/adni_label_all_large.npy', subj_label_all)
np.save('/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/adni_date_interval_large.npy', subj_date_interval)



pdb.set_trace()
