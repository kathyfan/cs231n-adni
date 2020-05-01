import os
import shutil
import numpy as np
import torch
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import scipy.ndimage
import scipy.stats
import scipy.misc as sci
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.color


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
    # adding loss given by intermediate timestep output, in mode: all ts predict final/future label
    if not config['classify_by_label_all'] and len(output) > 1 and config['cls_intermediate']:
        label_reshape = reshape_output_and_label(output[1], label)
        loss_cls_mid = loss_cls_fn(output[1], label_reshape).squeeze(-1)
        ts_weight = (torch.Tensor(config['cls_intermediate']) / config['cls_intermediate'][-1]).repeat(mask.shape[0],1).to(config['device'])
        loss_cls_mid = (loss_cls_mid * mask * ts_weight).mean()
        loss += config['lambda_mid'] * loss_cls_mid
        losses.append(loss_cls_mid)
    # adding l2-loss_regularization
    if config['regularizer']:
        loss_reg = loss_regularization_fn([model.fc1, model.fc2, model.fc3], config['regularizer'])
        loss += config['lambda_reg'] * loss_reg
        losses.append(loss_reg)
    return loss, losses

def vote_prediction(pred, mask):
    # pred_bi = np.where(pred>=0.5, 1, 0).squeeze(-1) * mask
    # pred_new = [1 if pred_bi[i,:].sum()>=0.5*mask[i,:].sum() else 0 for i in range(pred_bi.shape[0])]
    pred_new = (pred.squeeze(-1) * mask).sum(1) / mask.sum(1)
    return pred_new

def save_checkpoint(state, is_best, checkpoint_dir):
    print("saving checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        print("updating best checkpoint")
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

def load_pretrained_model(model, device, ckpt_path):
    print('load pretrained model from: ', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    #pdb.set_trace()

    if 'feature_extractor' in checkpoint.keys():
        model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    else:
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(ckpt_path, \
                checkpoint['epoch'], checkpoint['monitor_metric']))
        pretrained_model_state = checkpoint['model']

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

def compute_result_stat(pred, label, num_cls, mask):
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
