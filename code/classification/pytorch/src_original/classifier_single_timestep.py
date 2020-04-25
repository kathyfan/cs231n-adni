import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np

from model import *
from utils import *

config = {}
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu = '0'
config['device'] = torch.device('cuda:'+ gpu)
# config['device'] = torch.device('cpu')
config['data_path'] = '/fs/neurosci01/qingyuz/ncanda/ncanda_structural_img/img_64'
config['csv_path'] = '/fs/neurosci01/visitors/jiahongo/ncanda-alcoholism/data/NCANDA_preprocessed.csv'
# config['data_path'] = '../data'
# config['csv_path'] = '../data/NCANDA_preprocessed.csv'
config['ckpt_path'] = '../ckpt/SingleTimestep3DCNN'
if not os.path.exists(config['ckpt_path']):
    os.mkdir(config['ckpt_path'])
config['ckpt_name'] = 'model_best.pth.tar'  # only for testing
config['img_size'] = (64, 64, 64)
config['cls_type'] = 'binary'
# config['cls_type'] = 'multiple'
config['batch_size'] = 2
config['num_fold'] = 4
config['fold'] = 0
config['oversample'] = True
config['oversample_ratios'] = [0.4, 0.6]
# config['oversample_ratios'] = [0.2, 0.3, 0.6, 0.6]
# config['oversample_ratios'] = None
config['loss_weighted'] = True
config['loss_ratios'] = [1, 2]
# config['loss_ratios'] = [1, 1, 2, 2]
# config['loss_ratios'] = None
config['shuffle'] = False

config['epochs'] = 30
config['continue_train'] = False

config['phase'] = 'train'
# config['phase'] = 'test'

trainData= Data(dataset_type='single_timestep', oversample=config['oversample'], oversample_ratios=config['oversample_ratios'],
        data_path=config['data_path'], csv_path=config['csv_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='train', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0)
config['num_cls'] = trainData.dataset.num_cls

testData = Data(dataset_type='single_timestep', oversample=False, oversample_ratios=None,
        data_path=config['data_path'], csv_path=config['csv_path'], img_size=config['img_size'], cls_type=config['cls_type'],
        set='test', num_fold=config['num_fold'], fold=config['fold'], batch_size=config['batch_size'], shuffle=False, num_workers=0)
# testData = trainData

# model and loss
input_img_size = (config['img_size'][0]//2, config['img_size'][1], config['img_size'][2])
model = SingleTimestep3DCNN(in_num_ch=1, img_size=input_img_size, inter_num_ch=16,
                            kernel_size=3, conv_act='relu', fc_act='tanh', num_cls=config['num_cls']).to(config['device'])

loss_cls_fn, pred_fn = define_loss_fn(data=trainData, num_cls=config['num_cls'], loss_weighted=config['loss_weighted'], loss_ratios=config['loss_ratios'])
# loss_cls_fn = loss_cls_fn.to(device)

def train(model, trainData, testData, loss_cls_fn, pred_fn, config):
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')   # dynamic change lr according to val_loss

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

    global_iter = 0
    monitor_loss_best = 100
    iter_per_epoch = len(trainData.loader)
    start_time = time.time()
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        pred_all = []
        label_all = []
        for iter, sample in enumerate(trainData.loader, 0):
            global_iter += 1
            # pdb.set_trace()

            img = sample['image'].to(config['device'], dtype=torch.float)
            label = sample['label'].to(config['device'], dtype=torch.long)
            if config['num_cls'] == 2:
                label = label.unsqueeze(1).type(torch.float)
            output = model(img)

            loss = 0
            loss_cls = loss_cls_fn(output, label)
            loss += loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred_fn(output)
            pred_all.append(pred.detach().cpu().numpy())
            label_all.append(label.cpu().numpy())

            if global_iter % 100 == 0:
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_cls: %.8f' % \
                        (epoch, iter, iter_per_epoch, time.time()-start_time, loss_cls.item()))

        # pdb.set_trace()
        info = 'epoch%02d_gbiter%06d' % (epoch, global_iter)

        print('Epoch: [%2d] Training Results' % (epoch))
        pred_all = np.concatenate(pred_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        stat = compute_result_stat(pred_all, label_all, config['num_cls'])
        print_result_stat(stat)
        save_result_stat(stat, config, info=info)

        print('Epoch: [%2d] Testing Results' % (epoch))
        monitor_loss = evaluate(model, testData, loss_cls_fn, pred_fn, config, info=info)
        scheduler.step(monitor_loss)

        # save ckp
        if monitor_loss_best > monitor_loss or epoch % 10 == 1 or epoch == config['epochs']-1:
            is_best = (monitor_loss_best > monitor_loss)
            monitor_loss_best = monitor_loss if is_best == True else monitor_loss_best
            state = {'epoch': epoch, 'monitor_loss': monitor_loss, \
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                    'model': model.state_dict()}
            save_checkpoint(state, is_best, config['ckpt_path'])

def test(model, testData, loss_cls_fn, pred_fn, config):
    if not os.path.exists(config['ckpt_path']):
        raise ValueError('Testing phase, no checkpoint folder')
    [model], _ = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    evaluate(model, testData, loss_cls_fn, pred_fn, config, info='Test')


def evaluate(model, testData, loss_cls_fn, pred_fn, config, info='Default'):
    model.eval()
    loss_cls_all = 0
    pred_all = []
    label_all = []
    for iter, sample in enumerate(testData.loader):
        img = sample['image'].to(config['device'], dtype=torch.float)
        label = sample['label'].to(config['device'], dtype=torch.long)
        if config['num_cls'] == 2:
            label = label.unsqueeze(1).type(torch.float)
        output = model(img)
        loss_cls_all += loss_cls_fn(output, label)
        pred = pred_fn(output)
        pred_all.append(pred.detach().cpu().numpy())
        label_all.append(label.cpu().numpy())
    loss_cls_mean = loss_cls_all / (iter + 1)
    print(info, loss_cls_mean)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    stat = compute_result_stat(pred_all, label_all, config['num_cls'])
    print_result_stat(stat)
    save_result_stat(stat, config, info='test')

    return loss_cls_mean

if config['phase'] == 'train':
    train(model, trainData, testData, loss_cls_fn, pred_fn, config)
else:
    test(model, testData, loss_cls_fn, pred_fn, config)
