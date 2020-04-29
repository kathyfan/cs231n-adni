import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np

from model import *
from utils import *

config = {}
# config['phase'] = 'test'
config['phase'] = 'train'

config['gpu'] = '0,1'
config['device'] = torch.device('cuda:'+ config['gpu'])
# config['device'] = torch.device('cpu')

# config['dataset_name'] = 'ncanda'
# config['data_path'] = '/fs/neurosci01/qingyuz/ncanda/ncanda_structural_img/img_64'
# config['csv_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/data/NCANDA_preprocessed.csv'
# config['meta_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/data/NCANDA_metadata_preprocessed_small.npy'
# config['meta_size'] = 9

config['dataset_name'] = 'adni'


# config['num_timestep'] = 1
config['num_timestep'] = 5
if config['num_timestep'] == 1:
    config['model_type'] = 'single_timestep'
else:
    config['model_type'] = 'multiple_timestep'

config['model_name'] = 'SingleTimestep3DCNN'
# config['model_name'] = 'MultipleTimestepConcat'
# config['model_name'] = 'MultipleTimestepConcatMultipleOutput'
# config['model_name'] = 'MultipleTimestepLSTM'
# config['model_name'] = 'MultipleTimestepLSTMAvgPool'
# config['model_name'] = 'MultipleTimestepLSTMMetadata'
# config['model_name'] = 'MultipleTimestepConcatMetadata'
# config['model_name'] = 'MultipleTimestepLSTMMultimodal'
# config['skip_missing'] = True   # only for lstm
config['skip_missing'] = False   # only for lstm
# config['augment_random_drop'] = True

config['meta_only'] = True if 'Metadata' in config['model_name'] else False

# config['info'] = 'cat_'
# config['info'] = 'view_'

config['ckpt_timelabel'] = '2020_1_27_13_4'
# config['ckpt_timelabel'] = None
if config['ckpt_timelabel'] and config['phase'] == 'test':
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
                '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

# config['ckpt_path'] = '../ckpt/' + 'test'
config['ckpt_path'] = os.path.join('../ckpt/', config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):
    os.makedirs(config['ckpt_path'])

config['ckpt_name'] = 'model_best.pth.tar'  # only for testing
# config['ckpt_name'] = 'epoch041.pth.tar'  # only for testing
config['img_size'] = (64, 64, 64)
config['cls_type'] = 'binary'
# config['cls_type'] = 'multiple'
config['batch_size'] = 32
config['num_fold'] = 5
config['fold'] = 0

config['oversample'] = False
config['oversample_ratios'] = [0.4, 0.6]
# config['oversample_ratios'] = [0.3, 0.7]
# config['oversample_ratios'] = [0.2, 0.3, 0.6, 0.6]
# config['oversample_ratios'] = None

config['loss_weighted'] = False
# config['loss_ratios'] = [1, 1.8]
config['loss_ratios'] = [1, 0.7]
# config['loss_ratios'] = [1, 1, 2, 2]
# config['loss_ratios'] = None
# config['focal_loss'] = True
config['focal_loss'] = False

config['shuffle'] = (not config['oversample'])
config['epochs'] = 60
config['continue_train'] = False
config['regularizer'] = 'l2'
# config['regularizer'] = None
# config['lambda_reg'] = 0.1
config['lambda_reg'] = 0.01
config['lambda_balance'] = 0
config['init_lstm'] = True
# config['clip_grad'] = False
config['clip_grad'] = True
config['clip_grad_value'] = 1.

config['cls_intermediate'] = [1.,1.,1.,1.,1.]    # in lstm should be in ascending etc
# config['cls_intermediate'] = None
config['lambda_mid'] = 0.8  # should be less than 1.

# config['lr'] = 0.0002
config['lr'] = 0.0005
# config['lr'] = 0.002
# config['lr_fe'] = 0.0002
config['static_fe'] = False

config['pretrained'] = True
config['pretrained_keras_path'] = ['/fs/neurosci01/qingyuz/3dcnn/ADNI/res_raw/encoder.h5', '/fs/neurosci01/qingyuz/3dcnn/ADNI/res_raw/classifier.h5']
# config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/ckpt/pretrained/pretrained_adni_large_all.pth.tar'
config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/ckpt/pretrained/pretrained_adni_fe1to3.pth.tar'
# config['pretrained_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/ckpt/SingleTimestep3DCNN/2020_1_20_16_17/model_best.pth.tar'

if config['phase'] == 'train':
    save_config_file(config)

trainData= Data(dataset_type=config['model_type'], num_timestep=config['num_timestep'], oversample=config['oversample'], oversample_ratios=config['oversample_ratios'],
        data_path=config['data_path'], csv_path=config['csv_path'], meta_path=config['meta_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='train', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0, meta_only=config['meta_only'])
config['num_cls'] = trainData.dataset.num_cls

valData = Data(dataset_type=config['model_type'], num_timestep=config['num_timestep'], oversample=False, oversample_ratios=None,
        data_path=config['data_path'], csv_path=config['csv_path'], meta_path=config['meta_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='val', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=False, num_workers=0, meta_only=config['meta_only'])

testData = Data(dataset_type=config['model_type'], num_timestep=config['num_timestep'], oversample=False, oversample_ratios=None,
        data_path=config['data_path'], csv_path=config['csv_path'], meta_path=config['meta_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='test', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=False, num_workers=0, meta_only=config['meta_only'])

# model
input_img_size = (config['img_size'][0]//2, config['img_size'][1], config['img_size'][2])
if config['model_name'] == 'SingleTimestep3DCNN':
    model = SingleTimestep3DCNN(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcat':
    model = MultipleTimestepConcat(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepConcatMultipleOutput':
    model = MultipleTimestepConcatMultipleOutput(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'], num_timestep=config['num_timestep']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepLSTM':
    model = MultipleTimestepLSTM(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm']).to(config['device'])
elif config['model_name'] == 'MultipleTimestepLSTMAvgPool':
    model = MultipleTimestepLSTMAvgPool(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls'],
                                num_timestep=config['num_timestep'], skip_missing=config['skip_missing'], init_lstm=config['init_lstm']).to(config['device'])
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
    # pdb.set_trace()
    # if hasattr(model, 'feature_extractor'):
    #     optimizer_fe = optim.Adam(model.feature_extractor.parameters(), lr=config['lr_fe'])
    if hasattr(model, 'feature_extractor') and config['static_fe']:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)   # dynamic change lr according to val_loss

    # continue training
    start_epoch = 0
    if config['continue_train']:
        ckpt_filenames = sorted(glob.glob(config['ckpt_path']+'/epoch*.pth.tar'))
        ckpt_last_filename = os.path.basename(ckpt_filenames[-1])
        [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model],
                                                                            config['ckpt_path'],
                                                                            ['optimizer', 'scheduler', 'model'],
                                                                            config['device'],
                                                                            ckpt_last_filename)
    elif config['pretrained']:
        if 'Metadata' not in config['model_name']:
            # transfer_model_from_keras_to_pytorch(model, config['pretrained_keras_path'], config['pretrained_path'])
            model = load_pretrained_model(model, config['device'], ckpt_path=config['pretrained_path'])

    global_iter = 0
    monitor_metric_best = 0
    iter_per_epoch = len(trainData.loader)
    start_time = time.time()
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        pred_all = []
        label_all = []
        loss_cls_all = 0
        loss_all = 0
        for iter, sample in enumerate(trainData.loader, 0):
            global_iter += 1
            # pdb.set_trace()

            img = sample['image'].to(config['device'], dtype=torch.float)
            label = sample['label'].to(config['device'], dtype=torch.long)
            mask = sample['mask'].to(config['device'], dtype=torch.float)
            metadata = sample['metadata'].to(config['device'], dtype=torch.float)

            if config['num_cls'] == 2:
                label = label.unsqueeze(1).type(torch.float)
            if 'Metadata' in config['model_name']:
                output = model(metadata, mask)
            elif  'Multimodal' in config['model_name']:
                output = model(img, metadata, mask)
            else:
                output = model(img, mask)   # output is a list, [final_output, intermediate_output]
            pred = pred_fn(output[0])

            loss, losses = compute_loss(model, loss_cls_fn, config, output, pred, label, mask)
            loss_cls_all += losses[0].item()
            loss_all += loss.item()

            optimizer.zero_grad()
            loss.backward()
            if config['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_value'])
            optimizer.step()

            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(label.cpu().numpy())

        loss_cls_mean = loss_cls_all / iter_per_epoch
        loss_mean = loss_all / iter_per_epoch

        # pdb.set_trace()
        info = 'epoch%03d_gbiter%06d' % (epoch, global_iter)

        print('Epoch: [%3d] Training Results' % (epoch))
        pred_all = np.concatenate(pred_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        stat = compute_result_stat(pred_all, label_all, config['num_cls'])
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
        if monitor_metric_best < monitor_metric or epoch % 10 == 1 or epoch == config['epochs']-1:
            is_best = (monitor_metric_best < monitor_metric)
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
    loss_cls_all = 0
    loss_all = 0
    pred_all = []
    pred_mean_all = []
    label_all = []
    with torch.no_grad():   # else, the memory explode during model(img)
        for half in ['left', 'right']:
            testData.dataset.set_half(half)
            for iter, sample in enumerate(testData.loader):
                img = sample['image'].to(config['device'], dtype=torch.float)
                label = sample['label'].to(config['device'], dtype=torch.long)
                mask = sample['mask'].to(config['device'], dtype=torch.float)
                metadata = sample['metadata'].to(config['device'], dtype=torch.float)
                if config['num_cls'] == 2:
                    label = label.unsqueeze(1).type(torch.float)
                if 'Metadata' in config['model_name']:
                    output = model(metadata, mask)
                elif  'Multimodal' in config['model_name']:
                    output = model(img, metadata, mask)
                else:
                    output = model(img, mask)
                pred = pred_fn(output[0])
                loss, losses = compute_loss(model, loss_cls_fn, config, output, pred, label, mask)
                loss_cls_all += losses[0].item()
                loss_all += loss.item()

                pred_all.append(pred.detach().cpu().numpy())
                label_all.append(label.cpu().numpy())

    loss_cls_mean = loss_cls_all / (2*(iter + 1))
    loss_mean = loss_all / (2*(iter + 1))
    print(info, loss_cls_mean, loss_mean)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    stat = compute_result_stat(pred_all, label_all, config['num_cls'])
    stat['loss_cls'] = loss_cls_mean
    stat['loss_all'] = loss_mean
    print_result_stat(stat)
    if info != 'Test':
        save_result_stat(stat, config, info=info)
    if info == 'Test' and config['cls_type'] == 'binary':
        save_prediction(pred_all, label_all, testData.dataset.label_raw, config)
    # if len(output) > 1:
    #     pred_mean_all = np.concatenate(pred_mean_all, axis=0)
    #     stat = compute_result_stat(pred_mean_all, label_all, config['num_cls'])
    #     print('/n print vote prediction')
    #     print_result_stat(stat)
    # metric = loss_cls_mean + 0.3 * np.abs(stat['sensitivity'][0]-stat['specificity'][0])
    # metric = loss_cls_mean
    # pdb.set_trace()
    metric = stat['accuracy'][0]

    return metric

if config['phase'] == 'train':
    train(model, trainData, valData, testData, loss_cls_fn, pred_fn, config)
    save_result_figure(config)
else:
    # save_result_figure(config)
    test(model, testData, loss_cls_fn, pred_fn, config)
