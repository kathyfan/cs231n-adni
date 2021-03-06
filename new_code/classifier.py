import os
from os import path
import time
import torch
import torch.optim as optim
import numpy as np

from model import *
from utils import *
from data import *

config = {}

### STATE ### 
config['phase'] = 'test'                                       # or: 'test'

### DEVICE ###
config['gpu'] = '0,1'
# config['device'] = torch.device('cuda:'+ config['gpu'])         # or: torch.device('cpu')
config['device'] = torch.device('cpu')

### MODEL ###
config['ckpt_timelabel'] = None
if config['ckpt_timelabel'] and config['phase'] == 'test':
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + \
                '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

config['ckpt_path'] = os.path.join('../ckpt/', time_label)
if not os.path.exists(config['ckpt_path']):
    os.makedirs(config['ckpt_path'])

### DATA PARAMETERS ###
config['img_size'] = (64, 64, 64)

### TEST PARAMETERS ###
config['ckpt_name'] = 'model_best.pth.tar'                      # only for testing ...      'epoch041.pth.tar'
# config['ckpt_name'] = 'epoch010.pth.tar'


### TRAIN PARAMETERS ###
config['batch_size'] = 32
config['num_fold'] = 5
config['epochs'] = 40 
config['regularizer'] = 'l2'                                    # can toggle
config['lambda_reg'] = 0.01                                     # controls how much to regularize
config['clip_grad'] = True
config['clip_grad_value'] = 1.
config['lr'] = 0.001  # can toggle

### NOT ACTIVELY USED / TURNED OFF ###
config['static_fe'] = False
config['pretrained'] = False
config['pretrained_path'] = None
config['continue_train'] = False

save_config_file(config)

# split data into train, val, test
if path.exists("test_label_aug.npy"):
    train_data = np.load("train_aug.npy")
    val_data = np.load("val_aug.npy")
    test_data = np.load("test_aug.npy")
    train_label = np.load("train_label_aug.npy")
    val_label = np.load("val_label_aug.npy")
    test_label = np.load("test_label_aug.npy")
else: 
    train_data, val_data, test_data, train_label, val_label, test_label = get_data()

train_data = np.reshape(train_data, (2048, 1, 64, 64, 64))
val_data = np.reshape(val_data, (512, 1, 64, 64, 64))
test_data = np.reshape(test_data, (512, 1, 64, 64, 64))
print("classifier.py: done loading data")
print("train_data shape: ", train_data.shape)

# model
input_img_size = (config['img_size'][0], config['img_size'][1], config['img_size'][2])
model = SingleTimestep3DCNN(in_num_ch=1, img_size=input_img_size, inter_num_ch=16, fc_num_ch=16,
                                conv_act='relu', fc_act='tanh').to(config['device'])

# choose loss and prediction functions
loss_cls_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
pred_fn = torch.nn.Sigmoid()

# define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)   # dynamic change lr according to val_loss


def train(model, train_data, train_label, val_data, val_label, config):

    if config['pretrained']:
        model = load_pretrained_model(model, config['device'], ckpt_path=config['pretrained_path'])

    monitor_metric_best = 0
    iter_per_epoch = len(train_data) // config['batch_size']
    print("iter_per_epoch: ", iter_per_epoch)

    # store initial loss from end of each epoch
    # provides a summary statistic of loss over epochs
    losses_total_epochs = []
    losses_data_epochs = []
    losses_reg_epochs = []

    learning_rates = []

    print("classifier.py: starting training")
    
    for epoch in range(0, config['epochs']):

        print("classifier.py: line 113, epoch ", epoch)

        start_time = time.time()
        model.train()

        pred_all = []
        label_all = []
        loss = 0 # used for backward pass
        losses_total = [] # total loss history
        loss_total = 0 # running total loss and used for average calculation
        losses_data = [] # data loss history
        loss_data = 0 # running data loss
        losses_reg = [] # regularization loss history
        loss_reg = 0 # running regularization loss
        
        for i in range(iter_per_epoch):                                              
            print("training (epoch, iter): ", epoch, i)
            # generate random permutation of training data and take first batch_size indices
            idx_perm = np.random.permutation(train_data.shape[0])
            idx = idx_perm[:config['batch_size']]

            train_imgs = train_data[idx]
            train_labels = train_label[idx]
            imgs = torch.from_numpy(train_imgs).to(config['device'], dtype=torch.float)
            labels = torch.from_numpy(train_labels).to(config['device'], dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass of model, and make predictions
            print("starting forward pass")
            scores = torch.squeeze(model(imgs))    # to make it be [32] instead of [32,1]
            pred = pred_fn(scores)
            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(labels.cpu().numpy())

            # compute loss from data and regularization, and record
            print("computing loss")
            dloss, rloss = compute_loss(model, loss_cls_fn, config, scores, labels)
            loss_data += dloss.item()
            loss_reg += rloss.item()
            loss = dloss + rloss                                # needs to be a tensor b/c calling .backward() on this
            loss_total += dloss.item() + rloss.item()

            losses_data.append(dloss.item())           
            losses_reg.append(rloss.item())
            losses_total.append(dloss.item()+rloss.item())    

            # backward pass
            print("starting backward pass")
            loss.backward()
            if config['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_value'])

            print("finished backward pass")

            # optimize
            optimizer.step()

        # mean of total loss, across iterations in this epoch
        loss_mean = loss_total / iter_per_epoch

        # store final losses from this epoch
        losses_data_epochs.append(losses_data[-1])
        losses_reg_epochs.append(losses_reg[-1])
        losses_total_epochs.append(losses_total[-1])

        info = 'epoch%03d' % (epoch)
        print('Training time for epoch %3d: %3d' % (epoch, time.time() - start_time))
        print('Epoch [%3d]: Training Results' % (epoch))
        pred_all = np.concatenate(pred_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        stat = compute_result_stat(pred_all, label_all)
        stat['loss_mean'] = loss_mean

        stat['pred_all'] = pred_all
        stat['label_all'] = label_all

        # these hold the losses of each iteration within this epoch
        # i.e. data here only includes data from the current epoch
        stat['losses_data_hist'] = losses_data
        stat['losses_reg_hist'] = losses_reg
        stat['losses_total_hist'] = losses_total

        # these hold the losses of only the last iteration of each epoch, up to the current epoch
        # i.e. data here contains data from all prior epochs as well
        stat['losses_data_epochs'] = losses_data_epochs
        stat['losses_reg_epochs'] = losses_reg_epochs
        stat['losses_total_epochs'] = losses_total_epochs

        # print training stats (note: does not include learning rate)
        print_result_stat(stat)

        # perform validation for this epoch. Note that the scheduler depends on validation results,
        # so we perform this step before calling the scheduler.
        # validation stats will be printed from inside evaluate()
        print('Epoch [%3d]: Validation Results' % (epoch))
        monitor_metric = evaluate(model, val_data, val_label, loss_cls_fn, pred_fn, config, info='val')
        monitor_metric = monitor_metric[0] # acc is first and only item in list
        scheduler.step(monitor_metric)
        lr = optimizer.param_groups[0]['lr']
        print('lr: ', lr)
        learning_rates.append(lr)
        stat['learning_rates'] = learning_rates # stat['learning_rates'] hold list of learning rates up to the current epoch
        save_result_stat(str(epoch), stat, config, info=info)

        # save ckp of either 1) best epoch 2) every 10th epoch 3) last epoch
        if monitor_metric_best < monitor_metric or epoch % 10 == 1 or epoch == config['epochs']-1:
            is_best = (monitor_metric_best < monitor_metric) # want: high monitor_metric
            if is_best:
                monitor_metric_best = monitor_metric
            state = {'epoch': epoch, 'monitor_metric': monitor_metric,
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'model': model.state_dict()}
            save_checkpoint(state, is_best, config['ckpt_path'])

def test(model, test_data, test_label, loss_cls_fn, pred_fn, config):
    # Retrieve the trained model to test on
    if not os.path.exists(config['ckpt_path']):
        raise ValueError('Testing phase, no checkpoint folder')
    # [model], _ = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    [model], _ = load_checkpoint_by_key([model], '../ckpt/2020_6_3_2_16/', ['model'], config['device'], config['ckpt_name'])

    evaluate(model, test_data, test_label, loss_cls_fn, pred_fn, config, info='Test')

def evaluate(model, test_data, test_label, loss_cls_fn, pred_fn, config, info='Default'):
    print("evaluating")
    model.eval()
    pred_all = []
    label_all = []

    iters = len(test_data) // config['batch_size']
    print("iters: ", iters)

    losses_total = []  # total loss history
    loss_total = 0  # running total loss
    losses_data = []  # data loss history
    loss_data = 0  # running data loss
    losses_reg = []  # regularization loss history
    loss_reg = 0  # running regularization loss
    with torch.no_grad():   # else, the memory explode during model(img)
        for i in range(iters):                                                          
            print("evaluating iter= ", i)

            # generate random permutation of data and take first batch_size indices
            idx_perm = np.random.permutation(test_data.shape[0])
            idx = idx_perm[:config['batch_size']]

            test_imgs = test_data[idx]
            test_labels = test_label[idx]
            imgs = torch.from_numpy(test_imgs).to(config['device'], dtype=torch.float)
            labels = torch.from_numpy(test_labels).to(config['device'], dtype=torch.float)

            # forward pass and make predictions
            scores = torch.squeeze(model(imgs))
            if i == 0: # avoid redundancy in printing
                print("scores.shape: ", scores.shape)
            pred = pred_fn(scores)

            # compute losses
            dloss, rloss = compute_loss(model, loss_cls_fn, config, scores, labels)
            loss_data += dloss.item()
            loss_reg += rloss.item()
            loss_total += dloss.item() + rloss.item()

            losses_data.append(dloss.item())
            losses_reg.append(rloss.item())
            losses_total.append(dloss.item() + rloss.item())

            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(labels.cpu().numpy())

    loss_mean = loss_total / iters
    print("loss_mean: ", info, loss_mean)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    stat = compute_result_stat(pred_all, label_all)
    stat['loss_mean'] = loss_mean
    stat['losses_data_hist'] = losses_data
    stat['losses_reg_hist'] = losses_reg
    stat['losses_total_hist'] = losses_total
    stat['pred_all'] = pred_all
    stat['label_all'] = label_all
    print_result_stat(stat)
    save_result_stat('val', stat, config, info=info)

    acc = stat['accuracy']
    return acc

if config['phase'] == 'train':
    train(model, train_data, train_label, val_data, val_label, config)
    save_result_figure(config, str(config['epochs'] - 1))
else:
    test(model, test_data, test_label, loss_cls_fn, pred_fn, config)
