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
config['num_timestep'] = 2
config['fold'] = 0

# single, NC vs AD, fold 1
config['model_name'] = 'SingleTimestep3DCNN'
config['cls_type'] = 'binary'
config['remove_cls'] = 'MCI'
config['classify_by_label_all'] = True
# fold 0
config['ckpt_timelabel'] = '2020_2_29_19_38'
config['ckpt_name'] = 'epoch074.pth.tar'
# fold 1
'''
config['ckpt_timelabel'] = '2020_2_29_19_40'
config['ckpt_name'] = 'epoch079.pth.tar'
# fold 2
config['ckpt_timelabel'] = '2020_2_29_20_25'
config['ckpt_name'] = 'epoch059.pth.tar'
# fold 3
config['ckpt_timelabel'] = '2020_2_29_20_27'
config['ckpt_name'] = 'epoch042.pth.tar'
# fold 4
#config['ckpt_timelabel'] = '2020_3_14_5_40'
#config['ckpt_name'] = 'epoch054.pth.tar'
'''


config['model_name'] = 'MultipleTimestepGRUFutureAvgPool'
config['cls_type'] = 'binary'
config['remove_cls'] = 'MCI'
# fold 0
config['ckpt_timelabel'] = '2020_4_13_1_56'
config['ckpt_name'] = 'epoch024.pth.tar'


'''
# multiple, NC vs AD
config['model_name'] = 'MultipleTimestepGRUAvgPool'
config['cls_type'] = 'binary'
config['remove_cls'] = 'MCI'
config['classify_by_label_all'] = True
# fold 0
config['ckpt_timelabel'] = '2020_3_15_4_27'
config['ckpt_name'] = 'epoch055.pth.tar'
# fold 1
config['ckpt_timelabel'] = '2020_3_2_1_4'
config['ckpt_name'] = 'epoch058.pth.tar'
# fold 2
config['ckpt_timelabel'] = '2020_3_4_9_51'
config['ckpt_name'] = 'epoch022.pth.tar'
# fold 3
config['ckpt_timelabel'] = '2020_3_15_4_29'
config['ckpt_name'] = 'epoch049.pth.tar'
# fold 4
#config['ckpt_timelabel'] = '2020_3_15_15_52'
#config['ckpt_name'] = 'epoch024.pth.tar'
'''





# GRUPool, NC vs AD, fold 1
'''
config['model_name'] = 'MultipleTimestepGRUAvgPool'
config['ckpt_timelabel'] = '2020_3_2_1_4'
config['ckpt_name'] = 'epoch058.pth.tar'
config['cls_type'] = 'binary'
config['remove_cls'] = 'MCI'
config['classify_by_label_all'] = True
'''

# GRUPoolDate, NC vs AD, fold 1
'''
config['model_name'] = 'MultipleTimestepGRUAvgPoolDate'
config['ckpt_timelabel'] = '2020_3_2_1_6'
config['ckpt_name'] = 'epoch059.pth.tar'
config['cls_type'] = 'binary'
config['remove_cls'] = 'MCI'
config['classify_by_label_all'] = True
'''

# GRUPool, sMCI vs pMCI, fold 1
'''
config['model_name'] = 'MultipleTimestepGRUAvgPool'
config['ckpt_timelabel'] = '2020_3_2_1_10'
config['ckpt_name'] = 'epoch034.pth.tar'
config['cls_type'] = 'binary'
config['remove_cls'] = 'NC+AD'
config['classify_by_label_all'] = False
'''

# single, NC vs AD vs MCI, fold 1
'''
config['model_name'] = 'SingleTimestep3DCNN'
config['ckpt_timelabel'] = '2020_3_1_17_20'
config['ckpt_name'] = 'epoch045.pth.tar'
config['cls_type'] = 'multiple'
config['remove_cls'] = None
config['classify_by_label_all'] = True
'''

# GRU, NC vs AD vs MCI, fold 1
'''
config['model_name'] = 'MultipleTimestepGRU'
config['ckpt_timelabel'] = '2020_2_29_23_36'
config['ckpt_name'] = 'epoch016.pth.tar'
config['cls_type'] = 'multiple'
config['remove_cls'] = None
config['classify_by_label_all'] = True
'''

# GRUPool, NC vs AD vs MCI, fold 1
'''
config['model_name'] = 'MultipleTimestepGRUAvgPool'
config['ckpt_timelabel'] = '2020_3_2_1_0'
config['ckpt_name'] = 'epoch027.pth.tar'
config['cls_type'] = 'multiple'
config['remove_cls'] = None
config['classify_by_label_all'] = True
'''


config['batch_size'] = 16
#config['fold'] = 1


# do not change
#-----------------------------------------------------------------
config['phase'] = 'analyze'
config['gpu'] = '0,1'
config['device'] = torch.device('cuda:'+ config['gpu'])
config['dataset_name'] = 'adni'
config['data_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/'
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
elif config['model_name'] == 'MultipleTimestepGRUFutureAvgPool':
        model = MultipleTimestepLSTMFutureAvgPool(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
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

    #pdb.set_trace()
    model.eval()

    # RNN need to be in training model to compute gradient
    def apply_RNN(m):
        if type(m) == nn.GRU or type(m) == nn.LSTM:
            m.train()
    model.apply(apply_RNN)

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
    grad_all = []
    subj_id_all = []
    img_all = []
    if True:
    #with torch.no_grad():   # else, the memory explode during model(img)
        for half in ['left']:
            testData.dataset.set_half(half)
            for iter, sample in enumerate(testData.loader):
                #pdb.set_trace()
                #if iter > 2:
                #    break
                img = sample['image'].to(config['device'], dtype=torch.float)
                label = sample['label'].to(config['device'], dtype=torch.long)
                label_ts = sample['label_all'].to(config['device'], dtype=torch.long)
                mask = sample['mask'].to(config['device'], dtype=torch.float)
                interval = sample['interval'].to(config['device'], dtype=torch.float)

                if 'Single' in config['model_name']:
                    img = img.view(-1, 1, 64, 64, 64)
                img.requires_grad_()

                #pdb.set_trace()
                '''
                def hook_func_1(module, grad_in, grad_out):
                    print('grad_in', grad_in.shape)
                    print('grad_out', grad_out.shape)
                input_layer = model.feature_extractor._modules['conv1'][0]
                input_layer.register_backward_hook(hook_func_1)
                model.register_backward_hook(hook_func_1)
                '''
                output = model(img, mask)
                model.zero_grad()
                if len(output) == 1:
                    output[0].sum().backward()
                    grad_ts = img.grad.view(mask.shape[0], mask.shape[1], img.shape[2], img.shape[3], img.shape[4])
                    grad_ts = grad_ts.detach().cpu().numpy()
                else:
                    grad_ts = []
                    for tidx in range(output[1].shape[1]):
                        #output[1][:,tidx].sum().backward()
                        #output[1][:,tidx].sum().backward(retain_graph=(tidx!=output[1].shape[1]-1))
                        #grad_ts.append(img.grad.data.unsqueeze(1))
                        grad = torch.autograd.grad(output[1][:, tidx].sum(), img, retain_graph=(tidx!=output[1].shape[1]-1), only_inputs=True)
                        grad_ts.append(grad[0].unsqueeze(1))
                        model.zero_grad()
                        #img.grad.data.zero_()
                        
                    grad_ts = torch.cat(grad_ts, dim=1).detach().cpu().numpy()

                pred = pred_fn(output[0])
                #pdb.set_trace()

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

                pred_all.append(pred.detach().cpu().numpy())
                label_all.append(label.cpu().numpy())
                label_ts_all.append(label_ts.cpu().numpy())
                mask_all.append(mask.detach().cpu().numpy())
                pred_ts_all.append(pred_ts.detach().cpu().numpy())
                interval_all.append(interval.cpu().numpy())
                grad_all.append(grad_ts)
                subj_id_all.append(sample['subj_id'])
                img_all.append(img.view(mask.shape[0], mask.shape[1], img.shape[2], img.shape[3], img.shape[4]).detach().cpu().numpy())

                if len(output) > 2:
                    feat_all.append(output[2].detach().cpu().numpy())
                '''if len(output) > 3:
                    pdb.set_trace()
                    interval_pred_all.append(output[3].detach().cpu().numpy())
                '''

    pdb.set_trace()
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)
    label_ts_all = np.concatenate(label_ts_all, axis=0)
    pred_ts_all = np.concatenate(pred_ts_all, axis=0)
    grad_all = np.concatenate(grad_all, axis=0)
    subj_id_all = np.concatenate(subj_id_all, axis=0)
    img_all = np.concatenate(img_all, axis=0)

    np.save('grad_adni_'+config['model_name']+'_fold'+str(config['fold'])+'_test.npy', {'grad':grad_all, 'pred':pred_all, 'pred_ts':pred_ts_all, 'label':label_all, 'label_ts':label_ts_all, 'mask':mask_all})
        
    # single ts
    pdb.set_trace()
    vis_path = os.path.join('../visualize/adni/',config['model_name'])
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    saliency_mean = np.zeros(img_all.shape[2:])
    for idx, subj_id in enumerate(subj_id_all):
        label_subj = ('AD' if int(label_all[idx])==1 else 'NC')
        vis_subj_path = os.path.join(vis_path, subj_id+'_'+label_subj)
        if not os.path.exists(vis_subj_path):
            os.makedirs(vis_subj_path)
        saliency_subj = np.abs(grad_all[idx])
        saliency_subj -= saliency_subj.min()
        saliency_subj /= saliency_subj.max()
        #saliency_subj *= 255.
        img_subj = img_all[idx] 
        if len(saliency_subj.shape) == 4:
            num_map = 1
            saliency_subj = np.expand_dims(saliency_subj, axis=0)
        else:
            num_map = saliency_subj.shape[1]
        for g_idx in range(num_map):
            if mask_all[idx, g_idx] == 0:
                break
            for t_idx in range(img_subj.shape[0]):
                if mask_all[idx, t_idx] == 0:
                    break
                saliency_mean += saliency_subj[g_idx,t_idx]
                for z_idx in range(saliency_subj.shape[-1]):
                    img_slice = img_subj[t_idx,:,:,z_idx]
                    saliency_slice = saliency_subj[g_idx,t_idx,:,:,z_idx]
                    save_path = os.path.join(vis_subj_path, 'ds'+str(g_idx)+'_dI'+str(t_idx)+'_z'+str(z_idx)+'.png')
                    overlay_saliency_map(img_slice, saliency_slice, save_path, show_both=True)
    
    saliency_mean /= saliency_mean.max()
    vis_subj_path = os.path.join(vis_path, 'mean')
    if not os.path.exists(vis_subj_path):
        os.makedirs(vis_subj_path)
    np.save(os.path.join(vis_subj_path, 'mean.npy'), saliency_mean)
    for z_idx in range(saliency_subj.shape[-1]):
        img_slice = img_subj[0,:,:,z_idx]
        saliency_slice = saliency_mean[:,:,z_idx]
        save_path = os.path.join(vis_subj_path, 'z'+str(z_idx)+'.png')
        overlay_saliency_map(img_slice, saliency_slice, save_path, show_both=True)
    '''
   # analyze for each timepoint, only for NC/AD (label never change)
    for ts in range(config['num_timestep']):
        mask_ts = mask_all[:, ts]
        num_case = mask_ts.sum()
        pred_ts = pred_ts_all[:, ts]
        if config['classify_by_label_all']:
            label_ts = label_ts_all[:, ts]
        else:
            label_ts = label_all
            label_ts[mask_ts==0] = -1
        num_dict = {}
        for la in np.unique(label_ts):
            if la < 0:
                continue
            num_dict[la] = (label_ts==la).sum()
        stat = compute_result_stat(pred_ts, label_ts, config['num_cls'], mask_ts)
        print(ts, ': Number of cases', int(num_case), num_dict)
        if config['num_cls'] == 2:
            print('balanced accuracy: ', stat['balanced_accuracy'], ', sensitivity: ', stat['sensitivity'], ', specificity: ', stat['specificity'])
        else:
            print('balanced accuracy: ', stat['balanced_accuracy'])
            print(stat['confusion_matrix'])
    '''

analyze(model, testData, loss_cls_fn, pred_fn, config)
