import os
import shutil
import numpy as np
import torch
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt


def save_config_file(config):
    file_path = os.path.join(config['ckpt_path'], 'config.txt')
    f = open(file_path, 'w')
    for key, value in config.items():
        f.write(key + ': ' + str(value) + '\n')
    f.close()

def loss_regularization_fn(layer_list, regularizer):
    rloss = 0
    for layer in layer_list:
        for weight in layer.parameters():
            if regularizer == 'l2':
                rloss += weight.norm()
            elif regularizer == 'l1':
                rloss += torch.mean(torch.abs(weight))
            else:
                raise ValueError('Regularizer not implemented')
    return rloss

# return data loss, regularization loss
def compute_loss(model, loss_cls_fn, config, output, labels):
    dloss = loss_cls_fn(outputs, labels)

    rloss = 0
    # add regularization loss, if specified by config
    if config['regularizer']:
        rloss = loss_regularization_fn([model.fc1, model.fc2, model.fc3], config['regularizer'])
        rloss *= config['lambda_reg']
    return dloss, rloss

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

# calculate statistics of prediction result
# note that downstream, we are currently only using accuracy
def compute_result_stat(pred, label):
    num_case = len(pred)
    stat = {}
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
    fpr, tpr, _ = sklearn.metrics.roc_curve(label, pred)
    stat['auc'] = sklearn.metrics.auc(fpr, tpr)

    return stat

def print_result_stat(stat):
    for key, value in stat.items():
        print(key, value)

def save_result_stat(idx, stat, config, info='Default'):
    stat_path = os.path.join(config['ckpt_path'], idx, 'stat.csv')
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
    columns = [col for col in stat.columns][2:]
    columns = sorted(columns)
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
