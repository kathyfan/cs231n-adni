import os
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import scipy.ndimage
import scipy.stats
import scipy.misc as sci
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.color
import h5py

import pdb

'''
classification dataset
'''

class LongitudinalDataset(Dataset):
    def __init__(self, data_train_path, data_test_path, label_path, label_all_path=None, cls_type='regression',
                num_timestep=3, set='train', num_fold=5, fold=0):
        self.set = set
        if self.set == 'train':
            self.data = h5py.File(data_train_path, 'r')
        else:
            self.data = h5py.File(data_test_path, 'r')
        self.label_dict = np.load(label_path).item()
        if label_all_path:
            self.label_all_dict = np.load(label_all_path).item()
        else:
            self.label_all_dict = None
            print('No label all file')
        #pdb.set_trace()
        self.cls_type = cls_type
        self.num_timestep = num_timestep
        self.num_fold = num_fold
        self.fold = fold


    def _load_selected_fold(self):
        self.num_subj_all = len(self.data)
        #subj_id_list, subj_label_list = self.label_dict.keys(), self.label_dict.values()
        #pdb.set_trace()
        subj_id_list, subj_label_list = zip(*sorted(zip(self.label_dict.keys(), self.label_dict.values())))
        subj_id_list = np.array(list(subj_id_list))
        subj_label_list = np.array(list(subj_label_list))

        if self.cls_type == 'binary':
            self.num_cls = 2
        elif self.cls_type == 'multiple':
            num_cls_ts = np.concatenate([np.unique(val) for key, val in self.label_all_dict.items()])
            self.num_cls = min(np.unique(np.array(subj_label_list)).shape[0], np.unique(num_cls_ts).shape[0])
        else:
            self.num_cls = 0

        #subj_id_list = subj_id_list[:50]
        #subj_label_list = subj_label_list[:50]
        #print(subj_id_list[:10])

        # first get 0.1val fold, the rest split 0.9*0.8=0.72train + 0.9*0.2=0.18test
        #pdb.set_trace()
        if self.num_cls > 2: # keep NC and AD cases same
            idx_old = (subj_label_list==0) + (subj_label_list==2)
            subj_id_old = subj_id_list[idx_old]
            subj_label_old = subj_label_list[idx_old]
            subj_id_new = subj_id_list[np.logical_not(idx_old)]
            subj_label_new = subj_label_list[np.logical_not(idx_old)]
            subj_id_select_old, subj_label_select_old = self.get_fold_idx(subj_id_old, subj_label_old)
            subj_id_select_new, subj_label_select_new = self.get_fold_idx(subj_id_new, subj_label_new)
            self.subj_id_select = np.concatenate([subj_id_select_old, subj_id_select_new])
            self.subj_label_select = np.concatenate([subj_label_select_old, subj_label_select_new])
        else:
            subj_id_select, subj_label_select = self.get_fold_idx(subj_id_list, subj_label_list)
            self.subj_id_select = subj_id_select
            self.subj_label_select = subj_label_select

        self.num_subj = len(self.subj_id_select)
        return self.num_subj, self.num_cls, self.subj_id_select, self.subj_label_select

    def get_fold_idx(self, subj_id_list, subj_label_list):
        skf_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        train_test_idx, val_idx = skf_val.split(subj_id_list, subj_label_list).__next__()
        #print(val_idx)
        if self.set == 'val':
            subj_id_select = subj_id_list[val_idx]
            subj_label_select = subj_label_list[val_idx]
        else:
            skf = StratifiedKFold(n_splits=self.num_fold, shuffle=True, random_state=0)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(subj_id_list[train_test_idx], subj_label_list[train_test_idx])):
                if fold_idx == self.fold:
                    if self.set == 'train':
                        set_idx = train_idx
                    else:
                        set_idx = test_idx
                    subj_id_select = subj_id_list[train_test_idx][set_idx]
                    subj_label_select = subj_label_list[train_test_idx][set_idx]
                    break
        return subj_id_select, subj_label_select

    def compute_data_stats(self):
        #pdb.set_trace()
        print('Number of class ', self.num_cls)
        print('Number of cases ', self.num_subj)
        if self.cls_type != 'regression':
            self.cls_ratio = []
            for i in range(self.num_cls):
                res = np.where(self.subj_label_select == i)
                self.cls_ratio.append(res[0].shape[0]/self.num_subj)
                print('Class', i, 'ratio', self.cls_ratio[-1])

    def __len__(self):
        return self.num_subj

    def set_half(self, half):
        self.half = half

    def pad_crop_timestep(self, img, label_all):
        #print(img.shape)
        #pdb.set_trace()
        mask = np.ones((self.num_timestep,))
        if img.shape[0] < self.num_timestep:
            mask[img.shape[0]:] = 0
            img_pad = np.zeros((self.num_timestep-img.shape[0],)+img.shape[1:])
            img = np.concatenate([img, img_pad], 0)
            if label_all.shape[0] != self.num_timestep:
                label_all = np.pad(label_all, (0, self.num_timestep - label_all.shape[0]), 'constant', constant_values=-1)
                #print(mask, label_all)
        elif img.shape[0] > self.num_timestep:
            # only keep first and last timestep
            if self.num_timestep == 2:
                img = np.concatenate([np.expand_dims(img[0],0), np.expand_dims(img[-1],0)], axis=0)
                label_all = np.array([label_all[0], label_all[-1]])
            else:
                img = img[:self.num_timestep, :, :, :]
                label_all = label_all[:self.num_timestep]

        #print(mask)
        #print('after', img.shape)
        return img, mask, label_all

    def augment_by_drop_timestep(self, img, mask, label_all, interval):
        # augment by random drop timepoint
        if self.set == 'train':
            if mask.sum() >=3 and np.random.rand() < 0.2:
                idx = int(mask.sum() - 1)
                mask[idx] = 0
                img[idx, :, :, :] = 0
                label_all[idx] = -1
                interval[idx] = 0
        return img, mask, label_all, interval

    def select_from_augmented_images(self, imgs):
        if self.set == 'train':
            idx = np.random.choice(imgs.shape[0])
            #print(idx)
            img = imgs[idx, :, :, :, :]
        else:
            if self.half == 'left':
                img = imgs[0, :, :, :, :]
            else:
                img = imgs[1, :, :, :, :]
        return img

    def get_interval(self, subj_id):
        if self.date_interval_dict:
            interval = np.array(self.date_interval_dict[subj_id])
            if interval.shape[0] > self.num_timestep:
                interval = interval[:self.num_timestep]
            else:
                interval = np.pad(interval, (0, self.num_timestep - interval.shape[0]), 'constant', constant_values=0)
        else:
            interval = np.zeros((self.num_timestep,))
        return interval

    def __getitem__(self, idx):
        subj_id = self.subj_id_select[idx]
        try:
            gender = 1 if self.gender_dict[subj_id]=='F' else 0
        except:
            gender = -1
        #print(subj_id)
        #with open('ncanda_subj_id.txt', 'a') as file:
        #    file.write(subj_id+'\n')

        label = self.label_dict[subj_id]
        #print(label)

        if self.label_all_dict:
            label_all = np.array(self.label_all_dict[subj_id])
        else:
            label_all = np.repeat(label, self.num_timestep)

        #if label != self.subj_label_select[idx]:
        #    print('----------', subj_id, label)

        imgs = self.data[subj_id]
        img = self.select_from_augmented_images(imgs)

        interval = self.get_interval(subj_id)
        if self.num_timestep == 1:
            idx = np.random.choice(img.shape[0], 1)
            #if self.set == 'train':
            #    idx = np.random.choice(img.shape[0], 1)
            #else:
            #    idx = [0]
            img = img[idx[0]:idx[0]+1,:,:,:]
            mask = 1
            # print(img.shape, label_all.shape, idx)
            try:
                label_all = label_all[idx[0]]
            except:
                label_all = label_all[0]
        else:
            img, mask, label_all = self.pad_crop_timestep(img, label_all)
            img, mask, label_all, interval = self.augment_by_drop_timestep(img, mask, label_all, interval)

        #print(interval, mask, label_all)

        sample = {'subj_id': subj_id, 'image': img, 'label': label, 'mask': mask, 'label_all':label_all, 'interval':interval, 'gender':gender}
        return sample


class ADNIDataset(LongitudinalDataset):
    def __init__(self, data_train_path, data_test_path, label_path, label_all_path=None, date_interval_path=None, cls_type='binary',
                num_timestep=4, set='train', num_fold=5, fold=0, remove_cls='MCI'):
        super().__init__(data_train_path=data_train_path, data_test_path=data_test_path, label_path=label_path, label_all_path=label_all_path, cls_type=cls_type,
                num_timestep=num_timestep, set=set, num_fold=num_fold, fold=fold)

        if date_interval_path:
            self.date_interval_dict = np.load(date_interval_path).item()
        else:
            self.date_interval_dict = None

        if not self.label_all_dict:
            raise ValueError('No label_all_dict for ADNI dataset')

        if self.cls_type == 'binary':
            self.label_dict, self.label_all_dict = self.multiple_to_binary(remove_cls=remove_cls)
        elif self.cls_type == 'multiple':
            self.modify_multiple_label()
        else: # regression
            self.label_all_dict = self.modify_regression_label()

        self.num_subj, self.num_cls, self.subj_id_select, self.subj_label_select = self._load_selected_fold()
        self.compute_data_stats()

    def modify_regression_label(self):
        for key in self.label_dict.keys():
            self.label_all_dict[key] = np.array(self.label_all_dict[key]) - 1.
        return self.label_all_dict

    def multiple_to_binary(self, remove_cls='MCI'):
        # label = [0,2,3,4], label_all=[0,1,2]
        remove_key = []
        #pdb.set_trace()
        for key in self.label_dict.keys():
            if len(list(set(self.label_all_dict[key]))) >= 3:
                remove_key.append(key)
                print('delete 0->1->2', key, self.label_dict[key], self.label_all_dict[key])
                continue
            if remove_cls == 'AD':
                if self.label_dict[key] == 2:    # remove AD
                    remove_key.append(key)
                elif self.label_dict[key] in [3,4]:
                    self.label_dict[key] = 1
            elif remove_cls == 'MCI':
                if self.label_dict[key] in [1,3,4]:    # remove MCI
                    remove_key.append(key)
                elif self.label_dict[key] == 2:
                    self.label_dict[key] = 1
                    self.label_all_dict[key] = np.repeat(1, len(self.label_all_dict[key]))
            elif remove_cls == 'NC+AD':
                if self.label_dict[key] in [0,2]:    # remove NC+AD, classify sMCI/pMCI
                    remove_key.append(key)
                elif self.label_dict[key] == 3:
                    self.label_dict[key] = 0
                elif self.label_dict[key] == 4:
                    self.label_dict[key] = 1
        print('Remove', len(remove_key), remove_cls)
        for key in remove_key:
            del self.label_dict[key]
            if self.label_all_dict:
                del self.label_all_dict[key]
        return self.label_dict, self.label_all_dict

    def modify_multiple_label(self):
        print('Notice: label order in label and label_all are different')
        #for key in self.label_dict_all.keys()

class LABDataset(LongitudinalDataset):
    def __init__(self, data_train_path, data_test_path, label_path, label_all_path=None, date_interval_path=None, cls_type='binary',
                num_timestep=4, set='train', num_fold=5, fold=0, remove_cls='H'):
        super().__init__(data_train_path=data_train_path, data_test_path=data_test_path, label_path=label_path, label_all_path=label_all_path, cls_type=cls_type,
                num_timestep=num_timestep, set=set, num_fold=num_fold, fold=fold)

        if date_interval_path:
            self.date_interval_dict = np.load(date_interval_path).item()
        else:
            self.date_interval_dict = None

        if self.cls_type == 'binary':
            self.label_dict, self.label_all_dict = self.multiple_to_binary(remove_cls=remove_cls)
        elif self.cls_type == 'multiple':
            raise ValueError('Do not support multi-class for LAB dataset')

        self.num_subj, self.num_cls, self.subj_id_select, self.subj_label_select = self._load_selected_fold()
        self.compute_data_stats()

    def multiple_to_binary(self, remove_cls='H'):
        remove_key = []
        #pdb.set_trace()
        self.label_all_dict = {}
        for key in self.label_dict.keys():
            if remove_cls == 'H':
                if self.label_dict[key] == 2:   #H
                    remove_key.append(key)
                elif self.label_dict[key] == 3:     #HE
                    self.label_dict[key] = 1
            elif remove_cls == 'E':
                if self.label_dict[key] == 1:   #E
                    remove_key.append(key)
                elif self.label_dict[key] in [2, 3]:     #H/HE
                    self.label_dict[key] = 1
            else:
                raise ValueError('Not support remove '+ remove_cls)
            self.label_all_dict[key] = np.repeat(self.label_dict[key], self.num_timestep)
        print('Remove', len(remove_key), remove_cls)
        for key in remove_key:
            del self.label_dict[key]
            del self.label_all_dict[key]
        return self.label_dict, self.label_all_dict


class NCANDADataset(LongitudinalDataset):
    def __init__(self, data_train_path, data_test_path, label_path, label_all_path=None, date_interval_path=None, gender_path=None, cls_type='regression',
                num_timestep=5, set='train', num_fold=5, fold=0, label_path_cls=None):
        super().__init__(data_train_path=data_train_path, data_test_path=data_test_path, label_path=label_path, label_all_path=label_all_path, cls_type=cls_type,
                num_timestep=num_timestep, set=set, num_fold=num_fold, fold=fold)

        if date_interval_path:
            self.date_interval_dict = np.load(date_interval_path).item()
        else:
            self.date_interval_dict = None
        if gender_path:
            self.gender_dict = np.load(gender_path).item()
        else:
            self.gender_dict = None

        if self.cls_type == 'binary':
            self.label_dict, self.label_all_dict = self.multiple_to_binary()
        elif self.cls_type == 'regression':
            self.label_dict_reg = self.make_regression_gt()
            if label_path_cls == None:
                raise ValueError('Not support regression split fold without class label')
            self.label_dict = np.load(label_path_cls).item()

        self.num_subj, self.num_cls, self.subj_id_select, self.subj_label_select = self._load_selected_fold()
        if self.cls_type == 'regression':
            self.label_dict = self.label_dict_reg
        self.compute_data_stats()

    def multiple_to_binary(self):
        remove_key = []
        for key in self.label_dict.keys():
            if self.label_dict[key] >= 1:    # [1,2,3]
                self.label_dict[key] = 1
            self.label_all_dict[key] = np.where(np.array(self.label_all_dict[key])>=1, 1, 0)
        return self.label_dict, self.label_all_dict

    def make_regression_gt(self):
        #pdb.set_trace()
        self.label_dict_reg = {}
        for key in self.label_dict.keys():
            if self.label_dict[key] > 0:
                self.label_dict_reg[key] = np.log(self.label_dict[key])
            else:
                self.label_dict_reg[key] = 0
        #pdb.set_trace()
        return self.label_dict_reg


'''
Class to manage all data
'''
class Data(object):
    def __init__(self, dataset_name='ncanda', dataset_type='single_timestep', num_timestep=1, oversample=False, oversample_ratios=None,
            data_path='', csv_path='', meta_path='', img_size=(64, 64, 64), cls_type='binary',
            set='train', num_fold=4, fold=0, batch_size=32, shuffle=False, num_workers=8, meta_only=False, others=[None]):
        if oversample and shuffle:
            raise ValueError('Cannot do shuffle and oversample sampler at the same time')
        if oversample and cls_type == 'regression':
            raise ValueError('Do not support sampler for regression label')

        if dataset_name == 'ncanda':
            data_train_path = os.path.join(data_path, 'ncanda_augmented_new.h5')
            data_test_path = os.path.join(data_path, 'ncanda_original_new.h5')
            if cls_type == 'regression':
                label_path = os.path.join(data_path, 'ncanda_label_LifeDrnk.npy')
                label_path_cls = os.path.join(data_path, 'ncanda_label.npy')
            else:
                label_path = os.path.join(data_path, 'ncanda_label.npy')
                label_path_cls = None
            label_all_path = os.path.join(data_path, 'ncanda_label_all.npy')
            gender_path = os.path.join(data_path, 'ncanda_gender.npy')
            date_interval_path = None
            self.dataset = NCANDADataset(data_train_path, data_test_path, label_path, label_all_path, date_interval_path, gender_path, cls_type, num_timestep, set, num_fold, fold, label_path_cls)
            #self.dataset =  LongitudinalDataset(data_path, csv_path, meta_path, img_size, num_timestep, cls_type, set, num_fold, fold, meta_only)

        elif dataset_name == 'adni':
            data_train_path = os.path.join(data_path, 'adni_augmented_large.h5')
            data_test_path = os.path.join(data_path, 'adni_original_large.h5')
            label_path = os.path.join(data_path, 'adni_label_large.npy')
            label_all_path = os.path.join(data_path, 'adni_label_all_large.npy')
            date_interval_path = os.path.join(data_path, 'adni_date_interval_large.npy')
            self.dataset = ADNIDataset(data_train_path, data_test_path, label_path, label_all_path, date_interval_path, cls_type, num_timestep, set, num_fold, fold, others[0])

        elif dataset_name == 'lab':
            data_train_path = os.path.join(data_path, 'lab_augmented.h5')
            data_test_path = os.path.join(data_path, 'lab_original.h5')
            label_path = os.path.join(data_path, 'lab_label.npy')
            label_all_path = None
            date_interval_path = os.path.join(data_path, 'lab_date_interval.npy')
            self.dataset = LABDataset(data_train_path, data_test_path, label_path, label_all_path, date_interval_path, cls_type, num_timestep, set, num_fold, fold, others[0])

        if oversample == False:
            self.sampler = None
        else:
            weights = self._compute_sampler_weights(oversample_ratios)
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, self.dataset.num_subj)

        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, sampler=self.sampler)

    def _compute_sampler_weights(self, oversample_ratios):
        if oversample_ratios == None:
            cls_ratio = np.array(self.dataset.cls_ratio)
            oversample_ratios = 1.0 / cls_ratio
            oversample_ratios[np.isfinite(oversample_ratios)==False] = 0
        elif len(oversample_ratios) != self.dataset.num_cls:
            raise ValueError('Wrong oversample ratio')
        print('Oversample ratio', oversample_ratios)
        weights = [oversample_ratios[int(label)] for label in self.dataset.label]
        return weights

def save_config_file(config):
    file_path = os.path.join(config['ckpt_path'], 'config.txt')
    f = open(file_path, 'w')
    for key, value in config.items():
        f.write(key + ': ' + str(value) + '\n')
    f.close()

'''
Define classification loss function
'''
def define_loss_fn(data, num_cls=2, loss_weighted=False, loss_ratios=None):
    # set weight
    if loss_weighted and num_cls == 0:
        raise ValueError('Not support weighted loss for regression')
    if loss_weighted == False:
        weight = 0.5 * np.ones((num_cls,))
    elif loss_ratios == None:
        weight = 1.0/np.array(data.dataset.cls_ratio)
        weight[np.isfinite(weight)==False] = 0
        weight = weight / weight.sum()
    else:
        if len(loss_ratios) != num_cls:
            raise ValueError('Wrong loss weight ratio')
        else:
            loss_ratios = np.array(loss_ratios)
            weight = loss_ratios / loss_ratios.sum()
    weight = torch.tensor(weight, dtype=torch.float).cuda()
    print('Weighted loss ratio', weight)

    # diff num_cls
    if num_cls == 0:
        loss_cls_fn = torch.nn.MSELoss()
        def weighted_mse_loss(pred, label):
            res = torch.mean((0.2+label)*(pred-label)**2)
            return res
        #loss_cls_fn = weighted_mse_loss
        #pred_fn = torch.nn.ReLU()    # no identity function, already all pos, thus ReLU=identity
        pred_fn = lambda x: x
    elif num_cls == 2:
        weight = 1.* weight[1] / weight[0]
        loss_cls_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
        pred_fn = torch.nn.Sigmoid()
    else:
        loss_cls_fn = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        # pred_fn = torch.nn.LogSoftmax(1)
        pred_fn = torch.nn.Softmax(-1)
    return loss_cls_fn, pred_fn

def loss_regularization_fn(layer_list, regularizer):
    los_reg = 0
    for layer in layer_list:
        for weight in layer.parameters():
            if regularizer == 'l2':
                los_reg += weight.norm()
            elif regularizer == 'l1':
                los_reg += torch.mean(torch.abs(weight))
            else:
                raise ValueError('No regularizer')
    return los_reg

def loss_consistency_fn(pred, mask, label, config):
    #pdb.set_trace()
    if config['num_cls'] <= 2:    # binary/regression
        pred = pred.squeeze(-1)
    else:    # multiclass, AD
        pred = pred[:,:,-1]
    bs, ts = mask.shape[0], mask.shape[1]
    loss_cons = 0
    if config['dataset_name'] == 'adni':
        if config['num_cls'] == 2:   # AD / pMCI
            overlook_mask = (label==1)
        else:    # AD and pMCI
            overlook_mask = (label==2) | (label==4)
        overlook_mask = overlook_mask[:,0,0].float()
    else:
        # print('define overlook mask')
        overlook_mask = torch.ones_like(label[:,0,0])
    # only consider AD/MCI with label=1
    for i in range(ts):
        for j in range(i+1, ts):
            loss_tpm = overlook_mask * torch.clamp((pred[:, j] - pred[:, i]) * torch.min(mask[:, j], mask[:, i]), max=0.)
            loss_cons += loss_tpm.sum()
    loss_cons /= ts
    return -loss_cons

def reshape_output_and_label(output, label):
    #if output.shape != label.shape:
        # output: (bs, ts, cls), lstm
    while(len(label.shape) != len(output.shape)):
        label = label.unsqueeze(-1)
    label = label.repeat(1, output.shape[1], 1)  # (bs, ts, cls)
    return label

def focal_loss_fn(pred, label, config):
    if config['cls_type'] != 'binary':
        raise ValueError('No implementation for focal loss multiple class')
    gamma = 2.
    eps = 1e-8
    if config['loss_weighted']:
        alpha = config['loss_ratios'][1] / config['loss_ratios'][0]
    else:
        alpha = 1.
    loss_neg = (1.-label) * torch.pow(pred, gamma) * torch.log(1.-pred+eps)
    loss_pos = label * alpha * torch.pow(1-pred, gamma) * torch.log(pred+eps)
    return -(loss_neg + loss_pos).mean()

def compute_loss_ordinary(model, output, pred, labels, mask, config):
    loss = 0
    losses = []
    label = labels[0]
    label_ts = labels[1]

    if config['num_cls'] <= 2:
        raise ValueError('Only support multi-class classification')

    loss_ratios = np.array(config['loss_ratios'])
    weight = torch.tensor(loss_ratios, dtype=torch.float).cuda()
    loss_cls_nc_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight[0], reduction='none')
    loss_cls_ad_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight[1], reduction='none')

    label_ts_nc = (label_ts==0).type(torch.float)
    label_ts_ad = (label_ts==2).type(torch.float)
    if len(output) > 1:
        loss_cls_nc = loss_cls_nc_fn(output[1][:,:,0][mask==1], label_ts_nc[mask==1]).mean()
        loss_cls_ad = loss_cls_ad_fn(output[1][:,:,1][mask==1], label_ts_ad[mask==1]).mean()
    else:
        loss_cls_nc = loss_cls_nc_fn(output[0][:,0], label_ts_nc).mean()
        loss_cls_ad = loss_cls_ad_fn(output[0][:,1], label_ts_ad).mean()

    loss_cls_final = 0.5 * (loss_cls_nc + loss_cls_ad)
    loss += loss_cls_final
    losses.append(loss_cls_final)

    # consistency
    if len(output) > 1 and config['lambda_consistent']:
        label_reshape = reshape_output_and_label(output[1], label)
        pred_ts_ad = torch.nn.functional.sigmoid(output[1][:,:,1:]).squeeze(-1)

        bs, ts = mask.shape[0], mask.shape[1]
        loss_cons = 0
        overlook_mask = (label_reshape==2) | (label_reshape==4)
        overlook_mask = overlook_mask[:,0,0].float()

        # only consider AD/MCI with label=1
        for i in range(ts):
            for j in range(i+1, ts):
                loss_tpm = overlook_mask * torch.clamp((pred_ts_ad[:, j] - pred_ts_ad[:, i]) * torch.min(mask[:, j], mask[:, i]), max=0.)
                loss_cons += loss_tpm.sum()
        loss_cons /= ts
        loss += config['lambda_consistent'] * loss_cons
        losses.append(loss_cons)

    if config['regularizer']:
        loss_reg = loss_regularization_fn([model.lstm, model.fc1, model.fc2, model.fc2_pool, model.fc3_1, model.fc3_2], config['regularizer'])
        loss += config['lambda_reg'] * loss_reg
        losses.append(loss_reg)
    return loss, losses


def compute_loss(model, loss_cls_fn, pred_fn, config, output, pred, labels, mask, interval):
    loss = 0
    losses = []
    label = labels[0]
    label_ts = labels[1]

    if len(output) > 1:
        pred_all = pred_fn(output[1])

    #pdb.set_trace()
    if config['focal_loss']:
        loss_cls_final = focal_loss_fn(pred, label, config)
    else:
        if not config['classify_by_label_all']:    #old mode
            loss_cls_final = loss_cls_fn(output[0], label).mean()
        else:    # new mode
            if len(output) > 1:
                if config['num_cls'] == 2:
                    output1_re = output[1]
                    label_ts_re = label_ts
                    mask_re = mask
                else:
                    output1_re = output[1].view(-1, output[0].shape[-1])    # (bs, ts, num_cls) -> (bs*ts, num_cls)
                    label_ts_re = label_ts.view(-1)
                    mask_re = mask.view(-1)
                loss_cls_final = loss_cls_fn(output1_re[mask_re==1], label_ts_re[mask_re==1]).mean()
            else:
                loss_cls_final = loss_cls_fn(output[0], label_ts).mean()
    loss += loss_cls_final
    losses.append(loss_cls_final)
    # adding date interval loss
    if 'Date' in config['model_name']:
        pred_interval = output[3]
        #pdb.set_trace()
        loss_interval = torch.nn.MSELoss()(pred_interval.squeeze(-1), interval)
        loss += config['lambda_interval'] * loss_interval
        losses.append(loss_interval)
    # adding loss given by intermediate timestep output, in mode: all ts predict final/future label
    if not config['classify_by_label_all'] and len(output) > 1 and config['cls_intermediate']:
        label_reshape = reshape_output_and_label(output[1], label)
        loss_cls_mid = loss_cls_fn(output[1], label_reshape).squeeze(-1)
        ts_weight = (torch.Tensor(config['cls_intermediate']) / config['cls_intermediate'][-1]).repeat(mask.shape[0],1).to(config['device'])
        loss_cls_mid = (loss_cls_mid * mask * ts_weight).mean()
        loss += config['lambda_mid'] * loss_cls_mid
        losses.append(loss_cls_mid)
    # adding consistency loss, later prediction worser than previous
    if len(output) > 1 and config['lambda_consistent'] > 0:
        #pdb.set_trace()
        label_reshape = reshape_output_and_label(output[1], label)
        loss_cons = loss_consistency_fn(pred_all, mask, label_reshape, config)
        loss += config['lambda_consistent'] * loss_cons
        losses.append(loss_cons)
    # adding l2-loss_regularization
    if config['regularizer']:
        if config['model_name'] == 'MultipleTimestepConcatMultipleOutput' and config['cls_intermediate']:
            loss_reg = loss_regularization_fn([model.fc1, model.fc2, model.fc3, model.fc4], config['regularizer'])
        elif config['model_name'] == 'MultipleTimestepLSTM' or config['model_name'] == 'MultipleTimestepGRU':
            loss_reg = loss_regularization_fn([model.lstm, model.fc1, model.fc2, model.fc3], config['regularizer'])
        elif config['model_name'] == 'MultipleTimestepLSTMAvgPool' or config['model_name'] == 'MultipleTimestepGRUAvgPool':
            loss_reg = loss_regularization_fn([model.lstm, model.fc1, model.fc2, model.fc2_pool, model.fc3], config['regularizer'])
        elif config['model_name'] == 'MultipleTimestepLSTMMetadata':
            loss_reg = loss_regularization_fn([model.lstm, model.fc1, model.fc3], config['regularizer'])
        elif config['model_name'] == 'MultipleTimestepConcatMetadata':
            loss_reg = loss_regularization_fn([model.lstm, model.fc1, model.fc3], config['regularizer'])
        elif config['model_name'] == 'MultipleTimestepLSTMMultimodal':
            loss_reg = loss_regularization_fn([model.lstm, model.fc1, model.fc1_meta, model.fc2, model.fc3], config['regularizer'])
        else:
            loss_reg = loss_regularization_fn([model.fc1, model.fc2, model.fc3], config['regularizer'])
        loss += config['lambda_reg'] * loss_reg
        losses.append(loss_reg)
    # adding balance loss to make sensitivity=specificity
    if config['lambda_balance'] > 0:
        # pdb.set_trace()
        pred_bi = torch.where(pred>=torch.tensor(0.5).to(config['device']), torch.tensor(1.).to(config['device']), torch.tensor(0.).to(config['device']))
        tp = ((pred_bi == label) & (label == 1)).sum().float()
        tn = ((pred_bi == label) & (label == 0)).sum().float()
        fn = ((pred_bi != label) & (label == 1)).sum().float()
        fp = ((pred_bi != label) & (label == 0)).sum().float()
        loss_balance = torch.abs(tp / (tp + fn) - tn / (tn + fp))
        loss += config['lambda_balance'] * loss_balance
        losses.append(loss_balance)
    return loss, losses

def vote_prediction(pred, mask):
    # pdb.set_trace()
    # pred_bi = np.where(pred>=0.5, 1, 0).squeeze(-1) * mask
    # pred_new = [1 if pred_bi[i,:].sum()>=0.5*mask[i,:].sum() else 0 for i in range(pred_bi.shape[0])]
    pred_new = (pred.squeeze(-1) * mask).sum(1) / mask.sum(1)
    return pred_new

def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        print("update best checkpoint")
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')

def load_checkpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            values[i].load_state_dict(checkpoint[key])
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, \
                epoch, checkpoint['monitor_metric']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch

def define_keras_model():
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Activation, Dense, Dropout, Flatten, UpSampling3D, Input, ZeroPadding3D, Lambda, Reshape
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Conv3D, MaxPooling3D
    from keras import regularizers

    ft_bank_baseline = 16
    latent_dim = 16
    L2_reg = 0.1

    input_image = Input(shape=(32,64,64,1), name='input_image')
    feature = Conv3D(ft_bank_baseline, activation='relu', kernel_size=(3, 3, 3),padding='same')(input_image)
    feature = BatchNormalization()(feature)
    feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

    feature = Conv3D(ft_bank_baseline*2, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
    feature = BatchNormalization()(feature)
    feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

    feature = Conv3D(ft_bank_baseline*4, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
    feature = BatchNormalization()(feature)
    feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

    feature = Conv3D(ft_bank_baseline*4, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
    feature = BatchNormalization()(feature)
    feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)

    feature_dense = Flatten()(feature)
    encoder = Model(input_image, feature_dense)

    input_feature_clf = Input(shape=(2048,), name='input_feature_dense')
    feature_clf = Dense(latent_dim*4, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(input_feature_clf)
    feature_clf = Dense(latent_dim*2, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
    prediction_score = Dense(1, name='prediction_score',kernel_regularizer=regularizers.l2(L2_reg))(feature_clf)
    classifier = Model(input_feature_clf, prediction_score)

    return encoder, classifier

def transfer_model_from_keras_to_pytorch_fe1to3(model, pretrained_encoder, ckpt_path, state_dict):
    # define keras model
    pretrained_encoder.load_weights(ckpt_path)
    for idx, layer in enumerate([model.feature_extractor.conv1, model.feature_extractor.conv2, model.feature_extractor.conv3]):
        # conv, weight: keras (3,3,3,1,16), pytorch ()
        layer[0].weight.data = torch.from_numpy(pretrained_encoder.layers[3*idx+1].get_weights()[0]).permute(4,3,0,1,2)
        # conv, bias: keras (16), pytorch (16)
        layer[0].bias.data = torch.from_numpy(pretrained_encoder.layers[3*idx+1].get_weights()[1])
        # bn, gamma: keras [gamma, beta, mean, std], pytorch [gamma, bias]
        layer[2].weight.data = torch.from_numpy(pretrained_encoder.layers[3*idx+2].get_weights()[0])
        layer[2].bias.data = torch.from_numpy(pretrained_encoder.layers[3*idx+2].get_weights()[1])
    state_dict['feature_extractor'] = model.feature_extractor.state_dict()
    return state_dict

def transfer_model_from_keras_to_pytorch_fe1to4(model, pretrained_encoder, ckpt_path, state_dict):
    # define keras model
    pretrained_encoder.load_weights(ckpt_path)
    for idx, layer in enumerate([model.feature_extractor.conv1, model.feature_extractor.conv2, model.feature_extractor.conv3, model.feature_extractor.conv4]):
        # conv, weight: keras (3,3,3,1,16), pytorch ()
        layer[0].weight.data = torch.from_numpy(pretrained_encoder.layers[3*idx+1].get_weights()[0]).permute(4,3,0,1,2)
        # conv, bias: keras (16), pytorch (16)
        layer[0].bias.data = torch.from_numpy(pretrained_encoder.layers[3*idx+1].get_weights()[1])
        # bn, gamma: keras [gamma, beta, mean, std], pytorch [gamma, bias]
        layer[2].weight.data = torch.from_numpy(pretrained_encoder.layers[3*idx+2].get_weights()[0])
        layer[2].bias.data = torch.from_numpy(pretrained_encoder.layers[3*idx+2].get_weights()[1])
    state_dict['feature_extractor'] = model.feature_extractor.state_dict()
    return state_dict

def transfer_model_from_keras_to_pytorch_fc1to2(model, pretrained_classifier, ckpt_path, state_dict):
    # define keras model
    pretrained_classifier.load_weights(ckpt_path)
    for idx, layer in enumerate([model.fc1, model.fc2]):
        # linear, weight: keras (2048, 64), pytorch (64, 2048)
        layer[0].weight.data = torch.from_numpy(pretrained_classifier.layers[idx+1].get_weights()[0]).permute(1,0)
        # conv, bias: keras (16), pytorch (16)
        layer[0].bias.data = torch.from_numpy(pretrained_classifier.layers[idx+1].get_weights()[1])
    state_dict['fc1'] = model.fc1.state_dict()
    state_dict['fc2'] = model.fc2.state_dict()
    return state_dict

def transfer_model_from_keras_to_pytorch(model, ckpt_path, save_ckpt_path):
    state_dict = {}
    pretrained_encoder, pretrained_classifier = define_keras_model()
    state_dict = transfer_model_from_keras_to_pytorch_fe1to3(model, pretrained_encoder, ckpt_path[0], state_dict)
    state_dict = transfer_model_from_keras_to_pytorch_fc1to2(model, pretrained_classifier, ckpt_path[1], state_dict)
    torch.save(state_dict, save_ckpt_path)

def load_pretrained_model(model, device, ckpt_path):
    print('load pretrained model from: ', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    #pdb.set_trace()

    if 'feature_extractor' in checkpoint.keys():
        model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    # if 'fc1' in checkpoint.keys():
    #     model.fc1.load_state_dict(checkpoint['fc1'])
    # if 'fc2' in checkpoint.keys():
    #     model.fc2.load_state_dict(checkpoint['fc2'])
    else:
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(ckpt_path, \
                checkpoint['epoch'], checkpoint['monitor_metric']))
        #pdb.set_trace()
        pretrained_model_state = checkpoint['model']
        if 'LSTM' in ckpt_path or 'GRU' in ckpt_path:
            key_list = ['feature_extractor', 'fc1', 'fc2', 'fc2_pool', 'lstm']
        else:
            key_list = ['feature_extractor']

        def has_keywords(k, key_list):
            for key in key_list:
                if key in k:
                    return True
            return False
        pretrained_model_state = {k: v for k, v in pretrained_model_state.items() if has_keywords(k, key_list)}
        #pretrained_model_state = {k: v for k, v in pretrained_model_state.items() if 'feature_extractor' in k}
        model_state = model.state_dict()
        model_state.update(pretrained_model_state)
        model.load_state_dict(model_state)

    return model

def compute_result_stat_ordinary(pred, label, num_cls, mask):
    pdb.set_trace()
    if num_cls <= 2:
        raise ValueError('Only support numti-class')
    stat = {}
    idx = (mask != 0)
    pred = pred[idx]
    label = label[idx]
    pred_nc_bi = np.where(pred[:, 0]>=0.5, 1, 0)
    pred_ad_bi = np.where(pred[:, 1]>=0.5, 1, 0)
    label_nc = (label==0)
    label_ad = (label==2)
    stat['bacc_nc'] = balanced_accuracy_score(label_nc, pred_nc_bi)
    stat['bacc_ad'] = balanced_accuracy_score(label_ad, pred_ad_bi)
    stat['balanced_accuracy'] = 0.5 * (stat['bacc_nc'] + stat['bacc_ad'])
    return stat


def compute_result_stat(pred, label, num_cls, mask):
    #pdb.set_trace()
    if pred.shape[0] == mask.shape[0]:
        idx = (mask != 0)
        pred = pred[idx]
        label = label[idx]
    num_case = len(pred)
    stat = {}
    if num_cls == 2:
        pred_bi = np.where(pred>=0.5, 1, 0)
        tp = ((pred_bi == label) & (label == 1)).sum()
        tn = ((pred_bi == label) & (label == 0)).sum()
        fn = ((pred_bi != label) & (label == 1)).sum()
        fp = ((pred_bi != label) & (label == 0)).sum()
        stat['accuracy'] = 1.* (tp + tn) / num_case
        stat['sensitivity'] = 1.* tp / (tp + fn)
        stat['specificity'] = 1.* tn / (tn + fp)
        stat['balanced_accuracy'] = balanced_accuracy_score(label, pred_bi)
        stat['precision'] = 1.* tp / (tp + fp)
        stat['f1'] = 2.* tp / (2.*tp + fp + fn)
        #pdb.set_trace()
        fpr, tpr, _ = sklearn.metrics.roc_curve(label, pred)
        stat['auc'] = sklearn.metrics.auc(fpr, tpr)
    elif num_cls == 0:
        stat['mse'] = ((pred - label)**2).mean()
        _, _, stat['correlation_coefficient'], _, _ = scipy.stats.linregress(pred.reshape(num_case,), label.reshape(num_case,))
    else:
        #pdb.set_trace()
        pred_bi = pred.argmax(-1)
        stat['accuracy'] = 1.* (pred_bi == label).sum() / num_case
        stat['balanced_accuracy'] = balanced_accuracy_score(label, pred_bi)
        stat['confusion_matrix'] = confusion_matrix(label, pred_bi).reshape(num_cls**2)

    return stat

def print_result_stat(stat):
    for key, value in stat.items():
        print(key, value)

def save_result_stat(stat, config, info='Default'):
    stat_path = os.path.join(config['ckpt_path'], 'stat.csv')
    columns=['info',] + sorted(stat.keys())
    if not os.path.exists(stat_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(stat_path, mode='a', header=True)

    stat['info'] = info
    for key, value in stat.items():
        stat[key] = [value]
    df = pd.DataFrame.from_dict(stat)
    df = df[columns]
    df.to_csv(stat_path, mode='a', header=False)

def save_prediction(pred, label, label_raw, config):
    if label.shape[0] != label_raw.shape[0]:
        label_raw = np.tile(label_raw, 2)
    pred_bi = np.where(pred>=0.5, 1, 0)
    res = np.zeros((4,2))
    for i in range(4):
        pred_select = pred_bi[np.where(label_raw==i)]
        res[i,1] = pred_select.sum()
        res[i,0] = pred_select.shape[0] - res[i,1]
    print('Predication Stat for each class')
    print(res)


def save_result_figure(config):
    stat_path = os.path.join(config['ckpt_path'], 'stat.csv')
    stat = pd.read_csv(stat_path)
    data_train, data_test = [], []
    columns = [col for col in stat.columns][2:]
    columns = sorted(columns)
    # pdb.set_trace()
    df_train = stat.loc[(stat['info'] != 'test') & (stat['info'] != 'val')]
    df_test = stat.loc[stat['info'] == 'test']
    df_val = stat.loc[stat['info'] == 'val']
    epochs = min(df_train.shape[0], df_test.shape[0], df_val.shape[0])
    color = ['c', 'm', 'y', 'b', 'g', 'r']
    idx = -1
    linename = []
    for col in columns:
        if 'loss' in col:
            continue
        if col not in ['balanced_accuracy', 'mse', 'correlation_coefficient']:
            continue
        idx += 1
        plt.plot(range(1,epochs+1), df_train.loc[:,col], color=color[0])
        plt.plot(range(1,epochs+1), df_val.loc[:,col], color=color[1])
        plt.plot(range(1,epochs+1), df_test.loc[:,col], color=color[2])
        linename.extend([col+'_train', col+'_val', col+'_test'])
    plt.legend(linename, loc='lower right')
    plt.show()
    plt.savefig(os.path.join(config['ckpt_path'], 'stat.png'))
    plt.close()

    idx = -1
    linename = []
    for col in columns:
        if 'loss' not in col:
            continue
        idx += 1
        # stat_train = df_train.loc[:,col].to_numpy()
        # stat_test = df_test.loc[:,col].to_numpy()
        # loss_train = [float(stat_train[i][7:13]) for i in range(len(df_train))]
        # loss_test = [float(stat_test[i][7:13]) for i in range(len(df_test))]
        # plt.plot(range(1,epochs+1), loss_train, color=color[idx], linestyle='dashed')
        # plt.plot(range(1,epochs+1), loss_test, color=color[idx])
        plt.plot(range(1,epochs+1), df_train.loc[:,col], color=color[idx], linestyle='dashed')
        plt.plot(range(1,epochs+1), df_val.loc[:,col], color=color[idx], linestyle='dotted')
        plt.plot(range(1,epochs+1), df_test.loc[:,col], color=color[idx])
        linename.extend([col+'_train', col+'_val', col+'_test'])
    plt.legend(linename, loc='upper right')
    plt.show()
    plt.savefig(os.path.join(config['ckpt_path'], 'loss.png'))

def overlay_saliency_map(img, saliency, save_path, show_both=False):
    #pdb.set_trace()
    cmap = mpl.cm.get_cmap('jet')
    img -= img.min()
    img /= img.max()
    #img = np.clip(img, a_min=0)
    if show_both:
        img = np.concatenate([img, np.ones_like(img)], axis=1)
        saliency = np.concatenate([saliency, saliency], axis=1)
    saliency_rgba = cmap(saliency)
    background_hsv = skimage.color.rgb2hsv(np.dstack((img, img, img)))
    saliency_hsv = skimage.color.rgb2hsv(saliency_rgba[:,:,:3])
    background_hsv[..., 0] = saliency_hsv[..., 0]
    background_hsv[..., 1] = saliency_hsv[..., 1] * 0.5
    fusion = skimage.color.hsv2rgb(background_hsv)
    sci.imsave(save_path, fusion)
