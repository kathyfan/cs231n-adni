import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from model import *
from utils import *
from data import *

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
config['lambda_reg'] = 0.01                                     # controls how much to regularize
config['clip_grad'] = True
config['clip_grad_value'] = 1.
config['lr'] = 0.0005                                           # can toggle

### NOT ACTIVELY USED / TURNED OFF ###
config['static_fe'] = False
config['pretrained'] = False
config['pretrained_path'] = None
config['continue_train'] = False

save_config_file(config)

# split data into train, val, test
train, val, test = get_data('./subjects_idx.txt')
trainData = DataLoader(train, batch_size = config['batch_size'], shuffle=True)
valData = DataLoader(val, batch_size = config['batch_size'], shuffle=True)
testData = DataLoader(test, batch_size = config['batch_size'])

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
        loss = 0 # used for backward pass and average calculation
        losses_total = [] # total loss history
        loss_total = 0 # running total loss
        losses_data = [] # data loss history
        loss_data = 0 # running data loss
        losses_reg = [] # regularization loss history
        loss_reg = 0 # running regularization loss
        for i, sample in enumerate(trainData.loader, 0):
            iter += 1

            imgs, labels = sample

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(imgs)
            pred = pred_fn(outputs[0])

            # compute loss from data and regularization, and record
            dloss, rloss = compute_loss(model, loss_cls_fn, config, outputs, labels)
            loss_data += dloss.item()
            loss_reg += rloss.item()
            loss = dloss + rloss
            loss_total += dloss.item() + rloss.item()

            losses_data.append(dloss)
            losses_reg.append(rloss)
            losses_total.append(dloss+rloss)

            # backward pass
            loss.backward()
            if config['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_value'])

            # optimize
            optimizer.step()

            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(labels.cpu().numpy())

        # mean of total loss, across iterations in this epoch
        loss_mean = loss / iter_per_epoch

        info = 'epoch%03d_iter%06d' % (epoch, iter)
        print('Training time for epoch %3d: %3d' % (epoch, time.time() - start_time))
        print('Epoch: [%3d] Training Results' % (epoch))
        pred_all = np.concatenate(pred_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        stat = compute_result_stat(pred_all, label_all)
        stat['loss_mean'] = loss_mean
        print_result_stat(stat)
        save_result_stat(stat, config, info=info)

        # perform validation for this epoch. Note that the scheduler depends on validation results.
        print('Epoch: [%3d] Validation Results' % (epoch))
        monitor_metric = evaluate(model, valData, loss_cls_fn, pred_fn, config, info='val')
        scheduler.step(monitor_metric)
        print('lr: ', optimizer.param_groups[0]['lr'])

        # TODO: return and save validation loss

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
    pred_all = []
    label_all = []

    loss = 0  # used for backward pass and average calculation
    losses_total = []  # total loss history
    loss_total = 0  # running total loss
    losses_data = []  # data loss history
    loss_data = 0  # running data loss
    losses_reg = []  # regularization loss history
    loss_reg = 0  # running regularization loss
    with torch.no_grad():   # else, the memory explode during model(img)

        for iter, sample in enumerate(testData.loader):
            imgs = sample['image'].to(config['device'], dtype=torch.float)
            labels = sample['label'].to(config['device'], dtype=torch.long)
            output = model(imgs)
            pred = pred_fn(output[0])

            # compute losses
            dloss, rloss = compute_loss(model, loss_cls_fn, config, output, labels)
            loss_data += dloss.item()
            loss_reg += rloss.item()
            loss = dloss + rloss
            loss_total += dloss.item() + rloss.item()

            losses_data.append(dloss)
            losses_reg.append(rloss)
            losses_total.append(dloss + rloss)

            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(labels.cpu().numpy())

    loss_mean = loss / (2*(iter + 1))
    print(info, loss_mean)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    stat = compute_result_stat(pred_all, label_all, config['num_cls'])
    stat['loss_mean'] = loss_mean
    print_result_stat(stat)
    if info != 'Test': # validation phase
        save_result_stat(stat, config, info=info)
    else:
        save_prediction(pred_all, label_all, testData.dataset.label_raw, config)

    acc = stat['accuracy'][0]
    return acc

if config['phase'] == 'train':
    train(model, trainData, valData, loss_cls_fn, pred_fn, config)
    save_result_figure(config)
else:
    test(model, testData, loss_cls_fn, pred_fn, config)
