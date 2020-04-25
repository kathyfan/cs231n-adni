import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np

from model import *
from utils import *

config = {}
# change
#config['num_timestep'] = 1
config['num_timestep'] = 5

# single, NC vs AD, fold 1

config['model_name'] = 'SingleTimestep3DCNN'
# fold 0
#config['ckpt_timelabel'] = '2020_3_14_3_59'
#config['ckpt_name'] = 'epoch050.pth.tar'
# fold 1
#config['ckpt_timelabel'] = '2020_3_12_0_12'
#config['ckpt_name'] = 'epoch053.pth.tar'
# fold 2
#config['ckpt_timelabel'] = '2020_3_12_0_13'
#config['ckpt_name'] = 'epoch056.pth.tar'
# fold 3
#config['ckpt_timelabel'] = '2020_3_14_4_1'
#config['ckpt_name'] = 'epoch055.pth.tar'
# fold 4
#config['ckpt_timelabel'] = '2020_3_14_4_4'
#config['ckpt_name'] = 'epoch053.pth.tar'

# multiple, gru+pool, NC vs AD, fold 1

config['model_name'] = 'MultipleTimestepGRUAvgPool'
# fold 0
config['ckpt_timelabel'] = '2020_3_14_5_26'
config['ckpt_name'] = 'epoch016.pth.tar'
# fold 1
#config['ckpt_timelabel'] = '2020_3_15_0_45'
#config['ckpt_name'] = 'epoch017.pth.tar'
# fold 2
#config['ckpt_timelabel'] = '2020_3_15_0_47'
#config['ckpt_name'] = 'epoch022.pth.tar'
# fold 3
#config['ckpt_timelabel'] = '2020_3_15_0_51'
#config['ckpt_name'] = 'epoch016.pth.tar'
# fold 4
#config['ckpt_timelabel'] = '2020_3_15_0_52'
#config['ckpt_name'] = 'epoch017.pth.tar'


config['cls_type'] = 'binary'
config['remove_cls'] = None
config['classify_by_label_all'] = False

config['batch_size'] = 16
config['fold'] = 0


# do not change
#-----------------------------------------------------------------
config['phase'] = 'analyze'
config['gpu'] = '0,1'
config['device'] = torch.device('cuda:'+ config['gpu'])
config['dataset_name'] = 'ncanda'
config['data_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/NCANDA/'
config['model_type'] = 'multiple_timestep'
config['csv_path'] = None
config['meta_path'] = None
config['meta_size'] = 0
config['fe_arch'] = 'ehsan' # baseline, resnet_small, resnet
config['skip_missing'] = True   # only for lstm
config['meta_only'] = False
config['continue_train'] = False
config['img_size'] = (64, 64, 64)
config['num_fold'] = 5
config['oversample'] = False
config['oversample_ratios'] = None
config['loss_weighted'] = False
config['loss_ratios'] = None
config['focal_loss'] = False
config['shuffle'] = False
config['epochs'] = 60
config['regularizer'] = 'l2'
config['lambda_reg'] = 0.02
config['lambda_balance'] = 0
config['init_lstm'] = True
config['clip_grad'] = False
config['clip_grad_value'] = 1.
config['cls_intermediate'] = None
config['lambda_mid'] = 0.8  # should be less than 1.
config['lambda_consistent'] = 0
config['lr'] = 0.0005
config['static_fe'] = False
config['pretrained'] = False
config['pretrained_keras_path'] = ['/fs/neurosci01/qingyuz/3dcnn/ADNI/res_raw/encoder.h5', '/fs/neurosci01/qingyuz/3dcnn/ADNI/res_raw/classifier.h5']
config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_2_29_19_40/epoch054.pth.tar'    # fold 1

if config['ckpt_timelabel'] and (config['phase'] != 'train' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
                '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):
    os.makedirs(config['ckpt_path'])
#-----------------------------------------------------------------
# start here

if config['phase'] == 'train':
    save_config_file(config)

# data
print('Building Training Dataset')
trainData= Data(dataset_name=config['dataset_name'], dataset_type=config['model_type'], num_timestep=config['num_timestep'], oversample=config['oversample'], oversample_ratios=config['oversample_ratios'],
        data_path=config['data_path'], csv_path=config['csv_path'], meta_path=config['meta_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='train', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0, meta_only=config['meta_only'], others=[config['remove_cls']])
config['num_cls'] = trainData.dataset.num_cls

print('Building Validation Dataset')
valData = Data(dataset_name=config['dataset_name'], dataset_type=config['model_type'], num_timestep=config['num_timestep'], oversample=False, oversample_ratios=None,
        data_path=config['data_path'], csv_path=config['csv_path'], meta_path=config['meta_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='val', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=False, num_workers=0, meta_only=config['meta_only'], others=[config['remove_cls']])

print('Building Testing Dataset')
testData = Data(dataset_name=config['dataset_name'], dataset_type=config['model_type'], num_timestep=config['num_timestep'], oversample=False, oversample_ratios=None,
        data_path=config['data_path'], csv_path=config['csv_path'], meta_path=config['meta_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='test', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=False, num_workers=0, meta_only=config['meta_only'], others=[config['remove_cls']])

# model
input_img_size = config['img_size']
if config['model_name'] == 'SingleTimestep3DCNN':
    model = SingleTimestep3DCNN(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcat':
    model = MultipleTimestepConcat(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcatMultipleOutput':
    model = MultipleTimestepConcatMultipleOutput(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepLSTM':
    model = MultipleTimestepLSTM(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm'], rnn_type='LSTM', fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepGRU':
    model = MultipleTimestepLSTM(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm'], rnn_type='GRU', fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepLSTMAvgPool':
    model = MultipleTimestepLSTMAvgPool(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm'], rnn_type='LSTM', fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepGRUAvgPool':
    model = MultipleTimestepLSTMAvgPool(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm'], rnn_type='GRU', fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepGRUAvgPoolOrdinary':
    model = MultipleTimestepLSTMAvgPoolOrdinary(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm'], rnn_type='GRU', fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepGRUAvgPoolDate':
    model = MultipleTimestepLSTMAvgPoolDate(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm'], rnn_type='GRU', fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcatMetadata':
    model = MultipleTimestepConcatMetadata(in_num_ch=1, meta_size=config['meta_size'], fc_num_ch=16, fc_act='relu', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepLSTMMetadata':
    model = MultipleTimestepLSTMMetadata(in_num_ch=1, meta_size=config['meta_size'], fc_num_ch=16, fc_act='relu', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepLSTMMultimodal':
    model = MultipleTimestepLSTMMultimodal(in_num_ch=1, img_size=input_img_size, meta_size=config['meta_size'], inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm']).to(config['device'])
else:
    raise ValueError('The model is not implemented')

# loss
loss_cls_fn, pred_fn = define_loss_fn(data=trainData, num_cls=config['num_cls'], loss_weighted=config['loss_weighted'], loss_ratios=config['loss_ratios'])

def analyze(model, testData, loss_cls_fn, pred_fn, config):
    if not os.path.exists(config['ckpt_path']):
        raise ValueError('Testing phase, no checkpoint folder')
    [model], _ = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])

    model.eval()
    flag = False

    loss_cls_all = 0
    loss_all = 0
    pred_all = []
    pred_mean_all = []
    pred_ts_all = []
    label_ts_all = []
    feat_all = []
    mask_all = []
    label_all = []
    interval_all = []
    interval_pred_all = []
    gender_all = []
    fc2_reshape_all = []
    fc2_concat_all = []
    subj_id_all = []
    with torch.no_grad():   # else, the memory explode during model(img)
        for half in ['left']:
            testData.dataset.set_half(half)
            for iter, sample in enumerate(testData.loader):
                #pdb.set_trace()
                #if iter > 2:
                #    break
                img = sample['image'].to(config['device'], dtype=torch.float)
                # img[:,1] = img[:,0]
                # img[:,2] = img[:,0]
                # img[:,3] = img[:,0]
                label = sample['label'].to(config['device'], dtype=torch.long)
                label_ts = sample['label_all'].to(config['device'], dtype=torch.long)
                mask = sample['mask'].to(config['device'], dtype=torch.float)
                interval = sample['interval'].to(config['device'], dtype=torch.float)
                gender = sample['gender'].to(config['device'], dtype=torch.long)

                if 'Single' in config['model_name']:
                    img = img.view(-1, 1, 64, 64, 64)

                output = model(img, mask)
                pred = pred_fn(output[0])

                if config['num_cls'] == 2:
                    label = label.unsqueeze(1).type(torch.float)
                    label_ts = label_ts.unsqueeze(-1).type(torch.float)

                if 'Single' in config['model_name']:
                    if config['num_cls'] == 2:
                        pred_ts = pred.view(label_ts.shape)
                    else:
                        pred_ts = pred.view(label_ts.shape[0], label_ts.shape[1], config['num_cls'])

                if len(output) > 1:
                    pred_ts = pred_fn(output[1])

                # save fc2_reshape (before LP), fc2_concat (after LP before RNN)
                fc2_reshape_all.append(output[3].detach().cpu().numpy())
                fc2_concat_all.append(output[2].detach().cpu().numpy())
                gender_all.append(gender.cpu().numpy())

                pred_all.append(pred.detach().cpu().numpy())
                label_all.append(label.cpu().numpy())
                label_ts_all.append(label_ts.cpu().numpy())
                mask_all.append(mask.detach().cpu().numpy())
                pred_ts_all.append(pred_ts.detach().cpu().numpy())
                interval_all.append(interval.cpu().numpy())
                subj_id_all.append(sample['subj_id'])

                if len(output) > 2:
                    feat_all.append(output[2].detach().cpu().numpy())
                if len(output) > 3:
                    #pdb.set_trace()
                    interval_pred_all.append(output[3].detach().cpu().numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)
    label_ts_all = np.concatenate(label_ts_all, axis=0)
    pred_ts_all = np.concatenate(pred_ts_all, axis=0)

    fc2_reshape_all = np.concatenate(fc2_reshape_all, axis=0)
    fc2_concat_all = np.concatenate(fc2_concat_all, axis=0)
    gender_all = np.concatenate(gender_all, axis=0)
    subj_id_all = np.concatenate(subj_id_all, axis=0)
    
    np.save('feature_ncanda_fold'+str(config['fold'])+'_train.npy', {'subj_id': subj_id_all, 'fc2_reshape':fc2_reshape_all, 'fc2_concat':fc2_concat_all, 'gender':gender_all, 'pred':pred_all, 'pred_ts':pred_ts_all, 'label':label_all, 'label_ts':label_ts_all, 'mask':mask_all})
    
    pdb.set_trace()
    tpm_label_ts_all = np.tile(label_all, (1,5))
    tpm_label_ts_all[mask_all==0] = -1
    tpm_pred_ts_all = pred_ts_all.squeeze(-1)
    np.savetxt(config['ckpt_path']+'/pred_ts_all.csv', tpm_pred_ts_all, delimiter=',')
    np.savetxt(config['ckpt_path']+'/label_ts_all.csv', tpm_label_ts_all, delimiter=',')
    np.savetxt(config['ckpt_path']+'/mask_all.csv', mask_all, delimiter=',')
    pdb.set_trace()
    # analyze for each timepoint, only for NC/AD (label never change)
    for ts in range(config['num_timestep']):
        mask_ts = mask_all[:, ts]
        num_case = mask_ts.sum()
        pred_ts = pred_ts_all[:, ts]
        label_ts = tpm_label_ts_all[:, ts]
        num_dict = {}
        for la in np.unique(label_ts):
            if la < 0:
                continue
            num_dict[la] = ((label_ts.reshape(-1,)==la) & (mask_ts==1)).sum()
        stat = compute_result_stat(pred_ts, label_ts, config['num_cls'], mask_ts)
        print(ts, ': Number of cases', int(num_case), num_dict)
        if config['num_cls'] == 2:
            print('balanced accuracy: ', stat['balanced_accuracy'], ', sensitivity: ', stat['sensitivity'], ', specificity: ', stat['specificity'])
        else:
            print('balanced accuracy: ', stat['balanced_accuracy'])
            print(stat['confusion_matrix'])

    # same as above for cases with 4 or more ts
    idxes = (mask_all.sum(1) >= 4)
    for ts in range(config['num_timestep']):
        mask_ts = mask_all[idxes, ts]
        num_case = mask_ts.sum()
        pred_ts = pred_ts_all[idxes, ts]
        if config['classify_by_label_all']:
            label_ts = label_ts_all[idxes, ts]
        else:
            label_ts = label_all[idxes]
            label_ts[mask_ts==0] = -1
        num_dict = {}
        for la in np.unique(label_ts):
            if la < 0:
                continue
            num_dict[la] = ((label_ts.reshape(-1,)==la) & (mask_ts==1)).sum()
        stat = compute_result_stat(pred_ts, label_ts, config['num_cls'], mask_ts)
        print(ts, ': Number of cases', int(num_case), num_dict)
        if config['num_cls'] == 2:
            print('balanced accuracy: ', stat['balanced_accuracy'], ', sensitivity: ', stat['sensitivity'], ', specificity: ', stat['specificity'])
        else:
            print('balanced accuracy: ', stat['balanced_accuracy'])
            print(stat['confusion_matrix'])

    # pdb.set_trace()
    # analyze date interval prediction
    # if 'Date' in config['model_name']:
    #     interval_all = np.concatenate(interval_all, axis=0)
    #     interval_pred_all = np.concatenate(interval_pred_all, axis=0).squeeze(-1)
    #     loss_interval = ((interval_all - interval_pred_all)**2).mean()

    pdb.set_trace()
    # analyze feature difference
    if len(output) >= 2:
        # save feature results
        pred_path = os.path.join(config['ckpt_path'], 'pred.npy')
        feat_all = np.concatenate(feat_all, axis=0)
        save_dict = {'pred': pred_ts_all, 'mask': mask_all, 'feat': feat_all, 'label':label_all, 'label_all':label_ts_all}
        np.save(pred_path, save_dict)
        print('save prediction all')

        # compare feature difference
        num_subj = pred_ts_all.shape[0]
        ts = config['num_timestep']
        feat = feat_all
        mask = mask_all
        pred = pred_all

        feat_diff = np.zeros((num_subj, ts*(ts-1)//2, feat.shape[-1]))
        mask_diff = np.zeros((num_subj, ts*(ts-1)//2))
        idx = 0
        for i in range(ts):
            for j in range(i+1, ts):
                feat_diff[:, idx, :] = feat[:, i] - feat[:, j]
                mask_diff[:, idx] = np.minimum(mask[:, i], mask[:, j])
                idx += 1

        #pdb.set_trace()
        feat_diff = np.abs(feat_diff).mean(-1) * mask_diff
        idx = 0
        for i in range(ts):
            for j in range(i+1, ts):
                tpm = feat_diff[:, idx].sum() / (1.*mask_diff[:, idx].sum())
                print(i, j, tpm)
                idx += 1

        feat_diff_subj = np.zeros((200, feat.shape[-1]))
        for i in range(200):
            subj_idx = np.random.choice(num_subj, 2, replace=True)
            feat_diff_subj[i,:] = np.abs(feat[subj_idx[0], 0, :] - feat[subj_idx[1], 0, :])
        print('diff subj', feat_diff_subj.mean())


analyze(model, trainData, loss_cls_fn, pred_fn, config)
