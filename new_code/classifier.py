import os
import time
import torch
import torch.optim as optim
import numpy as np

from model import *
from utils import *

config = {}

### STATE ### 
config['phase'] = 'train'                                       # or: 'test'

### DEVICE ###
config['gpu'] = '0,1'
config['device'] = torch.device('cuda:'+ config['gpu'])         # or: torch.device('cpu')

### MODEL ###
config['ckpt_timelabel'] = None
if config['ckpt_timelabel'] and config['phase'] == 'test':
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
                '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

config['ckpt_path'] = os.path.join('../ckpt/', config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):
    os.makedirs(config['ckpt_path'])

### DATA PARAMETERS ###
config['img_size'] = (64, 64, 64)

### TEST PARAMETERS ###
config['ckpt_name'] = 'model_best.pth.tar'                      # only for testing ...      'epoch041.pth.tar'

### TRAIN PARAMETERS ###
config['batch_size'] = 32
config['num_fold'] = 5
config['epochs'] = 60
config['regularizer'] = 'l2'                                    # can toggle
config['lambda_reg'] = 0.01
config['clip_grad'] = True
config['clip_grad_value'] = 1.
config['lr'] = 0.0005                                           # can toggle
config['loss_weighted'] = False
config['loss_ratios'] = None

### NOT ACTIVELY USED / TURNED OFF ###
config['static_fe'] = False
config['pretrained'] = False
config['pretrained_path'] = None
config['continue_train'] = False

save_config_file(config)

# TODO: get data from data.py
trainData = None
valData = None
testData = None

# model
input_img_size = (config['img_size'][0], config['img_size'][1], config['img_size'][2])
model = SingleTimestep3DCNN(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                conv_act='relu', fc_act='tanh').to(config['device'])

# choose loss functions
loss_cls_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
pred_fn = torch.nn.Sigmoid()

# define optimizer and learnign rate scheduler
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)   # dynamic change lr according to val_loss


def train(model, trainData, valData, config):

    if config['pretrained']:
        model = load_pretrained_model(model, config['device'], ckpt_path=config['pretrained_path'])

    iter = 0 # iteration number; cumulative across epochs
    monitor_metric_best = 0
    iter_per_epoch = len(trainData.loader)
    for epoch in range(start_epoch+1, config['epochs']):
        start_time = time.time()
        model.train()
        pred_all = []
        label_all = []
        loss_cls_all = 0
        loss_all = 0
        for i, sample in enumerate(trainData.loader, 0):
            iter += 1

            img = sample['image'].to(config['device'], dtype=torch.float)
            label = sample['label'].to(config['device'], dtype=torch.long)

            if config['num_cls'] == 2:
                label = label.unsqueeze(1).type(torch.float)
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

        info = 'epoch%03d_iter%06d' % (epoch, iter)
        print('Training time for epoch %3d: %3d' % (epoch, time.time() - start_time))
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

        print('lr: ', optimizer.param_groups[0]['lr'])

        # save ckp of either 1) best epoch 2) every 10th epoch 3) last epoch
        if monitor_metric_best < monitor_metric or epoch % 10 == 1 or epoch == config['epochs']-1:
            is_best = (monitor_metric_best < monitor_metric) # want: high monitor_metric
            if is_best:
                monitor_metric_best = monitor_metric
            state = {'epoch': epoch, 'monitor_metric': monitor_metric, \
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                    'model': model.state_dict()}
            save_checkpoint(state, is_best, config['ckpt_path'])

def test(model, testData, loss_cls_fn, pred_fn, config):
    # Retrieve the trained model to test on
    if not os.path.exists(config['ckpt_path']):
        raise ValueError('Testing phase, no checkpoint folder')
    [model], _ = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    evaluate(model, testData, loss_cls_fn, pred_fn, config, info='Test')

def evaluate(model, testData, loss_cls_fn, pred_fn, config, info='Default'):
    model.eval()
    loss_cls_all = 0
    loss_all = 0
    pred_all = []
    label_all = []
    with torch.no_grad():   # else, the memory explode during model(img)
        for half in ['left', 'right']:
            testData.dataset.set_half(half)
            for iter, sample in enumerate(testData.loader):
                img = sample['image'].to(config['device'], dtype=torch.float)
                label = sample['label'].to(config['device'], dtype=torch.long)
                mask = sample['mask'].to(config['device'], dtype=torch.float)
                if config['num_cls'] == 2:
                    label = label.unsqueeze(1).type(torch.float)
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
    else:
        save_prediction(pred_all, label_all, testData.dataset.label_raw, config)

    metric = stat['accuracy'][0]
    return metric

if config['phase'] == 'train':
    train(model, trainData, valData, loss_cls_fn, pred_fn, config)
    save_result_figure(config)
else:
    test(model, testData, loss_cls_fn, pred_fn, config)
