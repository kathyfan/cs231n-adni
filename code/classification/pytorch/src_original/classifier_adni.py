import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np

from model import *
from utils import *

config = {}
#config['phase'] = 'test'
config['phase'] = 'train'

config['gpu'] = '0,1'
config['device'] = torch.device('cuda:'+ config['gpu'])
config['dataset_name'] = 'adni'
config['data_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/data/ADNI/'

config['csv_path'] = None
config['meta_path'] = None
config['meta_size'] = 0

config['num_timestep'] = 5
if config['num_timestep'] == 1:
    config['model_type'] = 'single_timestep'
else:
    config['model_type'] = 'multiple_timestep'

config['fe_arch'] = 'ehsan' # baseline, resnet_small, resnet
#config['model_name'] = 'SingleTimestep3DCNN'
#config['model_name'] = 'MultipleTimestepConcatMultipleOutput'
config['model_name'] = 'MultipleTimestepConcatMultipleOutputAvgPool'
#config['model_name'] = 'MultipleTimestepLSTM'
#config['model_name'] = 'MultipleTimestepGRU'
#config['model_name'] = 'MultipleTimestepLSTMAvgPool'
#config['model_name'] = 'MultipleTimestepGRUAvgPool'
#config['model_name'] = 'MultipleTimestepGRUAvgPoolDate'

config['skip_missing'] = True   # only for lstm

config['meta_only'] = True if 'Metadata' in config['model_name'] else False

config['continue_train'] = False

config['ckpt_timelabel'] = '2020_2_29_23_56'
# config['ckpt_timelabel'] = None
if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
                '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

# config['ckpt_path'] = '../ckpt/' + 'test'
config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):
    os.makedirs(config['ckpt_path'])

#config['ckpt_name'] = 'model_best.pth.tar'  # only for testing
config['ckpt_name'] = 'epoch059.pth.tar'  # only for testing
config['img_size'] = (64, 64, 64)
#config['cls_type'] = 'regression'
config['cls_type'] = 'binary'
#config['cls_type'] = 'multiple'
config['remove_cls'] = 'MCI'
#config['remove_cls'] = None
config['classify_by_label_all'] = True	# predict label_all from all each timestep, else predict label
config['batch_size'] = 16
config['num_fold'] = 5
config['fold'] = 2

config['oversample'] = False
config['oversample_ratios'] = [0.5, 0.35, 0.45]
# config['oversample_ratios'] = None

config['loss_weighted'] = True
#config['loss_ratios'] = [1.0, 0.8, 1.5]
#config['loss_ratios'] = [0.32, 0.2]    # NC vs MCI
#config['loss_ratios'] = [0.47, 0.53]    # NC vs AD, single ts
config['loss_ratios'] = [0.71, 0.87]    # NC vs AD, multiple ts
#config['loss_ratios'] = [1.18, 1., 1.44] # NC vs MCI vs AD = 866:1035:714
#config['loss_ratios'] = [1.18, 1.2, 1.44] # add weight on MCI, multiple ts
#config['loss_ratios'] = [1.93, 1.3, 2.17]  # 3-class, single ts
#config['loss_ratios'] = [1., 2.]   # sMCI vs pMCI

config['focal_loss'] = False

config['shuffle'] = (not config['oversample'])
config['epochs'] = 60

config['regularizer'] = 'l2'
#config['regularizer'] = None
config['lambda_reg'] = 0.02
config['lambda_balance'] = 0
config['init_lstm'] = True
config['clip_grad'] = False
config['clip_grad_value'] = 1.

config['cls_intermediate'] = [1.,1.,1.,1.,1.]    # in lstm should be in ascending etc
#config['cls_intermediate'] = None
config['lambda_mid'] = 0.8  # should be less than 1.

config['lambda_consistent'] = 0		# 1~2
config['lambda_interval'] = 0.02	# <0.05

config['lr'] = 0.0005
# config['lr'] = 0.0005
# config['lr'] = 0.002
config['static_fe'] = False

config['pretrained'] = True
config['pretrained_keras_path'] = ['/fs/neurosci01/qingyuz/3dcnn/ADNI/res_raw/encoder.h5', '/fs/neurosci01/qingyuz/3dcnn/ADNI/res_raw/classifier.h5']
# config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/ckpt/pretrained/pretrained_adni_large_all.pth.tar'
#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/pretrained/pretrained_adni_fe1to3.pth.tar'
#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_2_24_10_38/epoch092.pth.tar'    # fold 0
#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_2_23_23_44/epoch078.pth.tar'    # fold 2

#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_2_29_19_40/epoch054.pth.tar'    # single, NC vs AD, fold 1
#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/MultipleTimestepGRUAvgPool/2020_3_1_11_15/epoch035.pth.tar'    # multiple, NC vs AD, fold 1
#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_3_1_18_52/epoch039.pth.tar'    # single, sMCI vs pMCI, fold 1
#config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_3_1_17_20/epoch051.pth.tar'    # single, NC vs AD vs MCI, fold 1

config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/longitudinal_prediction/ckpt/adni/SingleTimestep3DCNN/2020_2_29_20_25/epoch043.pth.tar'    # single, NC vs AD, fold 1

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
#input_img_size = (config['img_size'][0]//2, config['img_size'][1], config['img_size'][2])
input_img_size = config['img_size']
if config['model_name'] == 'SingleTimestep3DCNN':
    model = SingleTimestep3DCNN(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcat':
    model = MultipleTimestepConcat(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep'], fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcatMultipleOutput':
    model = MultipleTimestepConcatMultipleOutput(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep'], fe_arch=config['fe_arch']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcatMultipleOutputAvgPool':
    model = MultipleTimestepConcatMultipleOutputAvgPool(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16, kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep'], fe_arch=config['fe_arch']).to(config['device'])
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


def train(model, trainData, valData, testData, loss_cls_fn, pred_fn, config):
    #pdb.set_trace()
    # if hasattr(model, 'feature_extractor'):
    #     optimizer_fe = optim.Adam(model.feature_extractor.parameters(), lr=config['lr_fe'])
    if hasattr(model, 'feature_extractor') and config['static_fe']:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5)   # dynamic change lr according to val_loss

    # continue training
    start_epoch = 0
    if config['continue_train']:
        #ckpt_filenames = sorted(glob.glob(config['ckpt_path']+'/epoch*.pth.tar'))
        #ckpt_last_filename = os.path.basename(ckpt_filenames[-1])
        ckpt_last_filename = config['ckpt_name']
        [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model],
                                                                            config['ckpt_path'],
                                                                            ['optimizer', 'scheduler', 'model'],
                                                                            config['device'],
                                                                            ckpt_last_filename)
    elif config['pretrained']:
        if 'Metadata' not in config['model_name']:
            # transfer_model_from_keras_to_pytorch(model, config['pretrained_keras_path'], config['pretrained_path'])
            model = load_pretrained_model(model, config['device'], ckpt_path=config['pretrained_path'])
            #[model.feature_extractor], _ = load_checkpoint_by_key([model.feature_extractor], config['pretrained_path'], ['model'], config['device'], config['ckpt_name'])

    global_iter = 0
    monitor_metric_best = 10
    iter_per_epoch = len(trainData.loader)
    start_time = time.time()
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        pred_all = []
        pred_ts_all = []
        label_ts_all = []
        label_all = []
        mask_all = []
        loss_cls_all = 0
        loss_all = 0
        for iter, sample in enumerate(trainData.loader, 0):
            global_iter += 1
            #pdb.set_trace()

            img = sample['image'].to(config['device'], dtype=torch.float)
            label = sample['label'].to(config['device'], dtype=torch.long)
            label_ts = sample['label_all'].to(config['device'], dtype=torch.long)
            mask = sample['mask'].to(config['device'], dtype=torch.float)
            interval = sample['interval'].to(config['device'], dtype=torch.float)
            #metadata = sample['metadata'].to(config['device'], dtype=torch.float)

            if config['num_cls'] <= 2:
                label = label.unsqueeze(1).type(torch.float)
                label_ts = label_ts.unsqueeze(-1).type(torch.float)
            if 'Metadata' in config['model_name']:
                output = model(metadata, mask)
            elif 'Multimodal' in config['model_name']:
                output = model(img, metadata, mask)
            else:
                output = model(img, mask)   # output is a list, [final_output, intermediate_output]
            pred = pred_fn(output[0])
            if config['classify_by_label_all'] and len(output) > 1:
                pred_ts = pred_fn(output[1])

            loss, losses = compute_loss(model, loss_cls_fn, pred_fn, config, output, pred, [label, label_ts], mask, interval)
            loss_cls_all += losses[0].item()
            loss_all += loss.item()

            optimizer.zero_grad()
            loss.backward()
            if config['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_value'])
            optimizer.step()

            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(label.cpu().numpy())
            label_ts_all.append(label_ts.cpu().numpy())
            mask_all.append(mask.cpu().numpy())
            if config['classify_by_label_all'] and len(output) > 1:
                pred_ts_all.append(pred_ts.detach().cpu().numpy())        

            print(losses[0])

        loss_cls_mean = loss_cls_all / iter_per_epoch
        loss_mean = loss_all / iter_per_epoch

        #pdb.set_trace()
        info = 'epoch%03d_gbiter%06d' % (epoch, global_iter)

        print('Epoch: [%3d] Training Results' % (epoch))
        pred_all = np.concatenate(pred_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        mask_all = np.concatenate(mask_all, axis=0)
        label_ts_all = np.concatenate(label_ts_all, axis=0).reshape(-1,)
        if config['classify_by_label_all'] and len(output) > 1:
            if config['num_cls'] <= 2:
                pred_ts_all = np.concatenate(pred_ts_all, axis=0).reshape(-1,)
            else:
                pred_ts_all = np.concatenate(pred_ts_all, axis=0).reshape(-1, config['num_cls'])
            stat = compute_result_stat(pred_ts_all, label_ts_all, config['num_cls'], mask_all.reshape(-1,))
        elif config['classify_by_label_all'] and len(output) == 1:
            stat = compute_result_stat(pred_all, label_ts_all, config['num_cls'], mask_all.reshape(-1,))
        else:
            stat = compute_result_stat(pred_all, label_all, config['num_cls'], mask_all.reshape(-1,))
        stat['loss_cls'] = loss_cls_mean
        stat['loss_all'] = loss_mean
        print_result_stat(stat)
        save_result_stat(stat, config, info=info)

        print('Epoch: [%3d] Validation Results' % (epoch))
        monitor_metric = evaluate(model, valData, loss_cls_fn, pred_fn, config, info='val')
        scheduler.step(monitor_metric)

        print('Epoch: [%3d] Testing Results' % (epoch))
        _ = evaluate(model, testData, loss_cls_fn, pred_fn, config, info='test')

        print('lr: ', optimizer.param_groups[0]['lr'])

        # save ckp
        if monitor_metric - monitor_metric_best <0.002  or True or epoch == config['epochs']-1:
            is_best = ((monitor_metric_best - monitor_metric) < 0.002)
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
            state = {'epoch': epoch, 'monitor_metric': monitor_metric, \
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                    'model': model.state_dict()}
            save_checkpoint(state, is_best, config['ckpt_path'])
            if is_best:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def test(model, testData, loss_cls_fn, pred_fn, config):
    if not os.path.exists(config['ckpt_path']):
        raise ValueError('Testing phase, no checkpoint folder')
    [model], _ = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    evaluate(model, testData, loss_cls_fn, pred_fn, config, info='Test')


def evaluate(model, testData, loss_cls_fn, pred_fn, config, info='Default'):
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
    with torch.no_grad():   # else, the memory explode during model(img)
        #for half in ['left', 'right']:
        for half in ['left']:
            testData.dataset.set_half(half)
            for iter, sample in enumerate(testData.loader):
                #pdb.set_trace()
                img = sample['image'].to(config['device'], dtype=torch.float)
                label = sample['label'].to(config['device'], dtype=torch.long)
                label_ts = sample['label_all'].to(config['device'], dtype=torch.long)
                mask = sample['mask'].to(config['device'], dtype=torch.float)
                interval = sample['interval'].to(config['device'], dtype=torch.float)
                #metadata = sample['metadata'].to(config['device'], dtype=torch.float)
                if config['num_cls'] <= 2:
                    label = label.unsqueeze(1).type(torch.float)
                    label_ts = label_ts.unsqueeze(-1).type(torch.float)
                if 'Metadata' in config['model_name']:
                    output = model(metadata, mask)
                elif  'Multimodal' in config['model_name']:
                    output = model(img, metadata, mask)
                else:
                    output = model(img, mask)
                pred = pred_fn(output[0])
                if len(output) > 1:
                    pred_ts = pred_fn(output[1])

                #pdb.set_trace()
                loss, losses = compute_loss(model, loss_cls_fn, pred_fn, config, output, pred, [label, label_ts], mask, interval)
                loss_cls_all += losses[0].item()
                loss_all += loss.item()

                pred_all.append(pred.detach().cpu().numpy())
                label_all.append(label.cpu().numpy())
                label_ts_all.append(label_ts.cpu().numpy())
                mask_all.append(mask.detach().cpu().numpy())
                if len(output) > 1:
                    pred_ts_all.append(pred_ts.detach().cpu().numpy())
                
                if flag:
                    feat_all.append(output[2].detach().cpu().numpy())

    loss_cls_mean = loss_cls_all / (iter + 1)
    loss_mean = loss_all / (iter + 1)
    print(info, loss_cls_mean, loss_mean)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)
    label_ts_all = np.concatenate(label_ts_all, axis=0).reshape(-1,)

    if config['classify_by_label_all'] and len(output) > 1:
        if config['num_cls'] <= 2:
            pred_ts_all = np.concatenate(pred_ts_all, axis=0).reshape(-1,)
        else:
            pred_ts_all = np.concatenate(pred_ts_all, axis=0).reshape(-1, config['num_cls'])
        stat = compute_result_stat(pred_ts_all, label_ts_all, config['num_cls'], mask_all.reshape(-1,))
    elif config['classify_by_label_all'] and len(output) == 1:
        stat = compute_result_stat(pred_all, label_ts_all, config['num_cls'], mask_all.reshape(-1,))
    else:
        stat = compute_result_stat(pred_all, label_all, config['num_cls'], mask_all.reshape(-1,))

    stat['loss_cls'] = loss_cls_mean
    stat['loss_all'] = loss_mean
    print_result_stat(stat)
    if info != 'Test':
        save_result_stat(stat, config, info=info)
    #if info == 'Test' and config['cls_type'] == 'binary':
    #    save_prediction(pred_all, label_all, testData.dataset.label_raw, config)
    if info == 'Test':
        #pdb.set_trace()
        num_all = pred_all.shape[0]
        pred_left = pred_all[:num_all//2]
        pred_right = pred_all[num_all//2:]
        pred_avg = 0.5*(pred_left + pred_right)
        pdb.set_trace()
        save_dict = {'pred':pred_all, 'pred_avg':pred_avg, 'label':label_all[:num_all//2], 'mask':mask_all[:num_all//2]}
        np.save(os.path.join(config['ckpt_path'], 'pred.npy'), save_dict)
        print('-------------final avg result:------------')
        stat = compute_result_stat(pred_avg, label_all[:num_all//2], config['num_cls'])
        print_result_stat(stat)
        num_ts_list = mask_all[:num_all//2].sum(1)
        label = label_all[:num_all//2]
        for num_ts in range(1, config['num_timestep']+1):
            #pdb.set_trace()
            pred_tpm = pred_avg[num_ts_list==num_ts]
            label_tpm = label[num_ts_list==num_ts]
            print(pred_tpm.shape[0])
            stat = compute_result_stat(pred_tpm, label_tpm, config['num_cls'])
            print('------', num_ts, '-------')
            print_result_stat(stat)
        
        
    
    # if len(output) > 1:
    #     pred_mean_all = np.concatenate(pred_mean_all, axis=0)
    #     stat = compute_result_stat(pred_mean_all, label_all, config['num_cls'])
    #     print('/n print vote prediction')
    #     print_result_stat(stat)
    # metric = loss_cls_mean + 0.3 * np.abs(stat['sensitivity'][0]-stat['specificity'][0])
    # metric = loss_cls_mean
    #metric = stat['accuracy'][0]

    '''try:
        metric = stat['balanced_accuracy'][0]
    except:
        metric = stat['balanced_accuracy']
    '''
    metric = loss_cls_mean 
   
    #pdb.set_trace()
    if flag:
        pred_path = os.path.join(config['ckpt_path'], 'pred.npy')
        pred_ts_all = np.concatenate(pred_ts_all, axis=0)
        mask_all = np.concatenate(mask_all, axis=0)
        feat_all = np.concatenate(feat_all, axis=0)
        save_dict = {'pred': pred_ts_all, 'mask': mask_all, 'feat': feat_all}
        np.save(pred_path, save_dict)
        print('save prediction all')

    return metric

if config['phase'] == 'train':
    train(model, trainData, valData, testData, loss_cls_fn, pred_fn, config)
    save_result_figure(config)
else:
    # save_result_figure(config)
    test(model, testData, loss_cls_fn, pred_fn, config)
