import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

import pdb

class FeatureExtractor(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu'):
        super(FeatureExtractor, self).__init__()

        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(inter_num_ch),
                        nn.MaxPool3d(2))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(2*inter_num_ch),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(0.1))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(4*inter_num_ch),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(0.1))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(inter_num_ch),
                        nn.MaxPool3d(2))


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4

'''
class FeatureExtractor_ResNet(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu', arch='resnet'):
        super(FeatureExtractor_ResNet, self).__init__()

        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1, stride=2),
                        conv_act_layer,
                        nn.BatchNorm3d(inter_num_ch))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        conv_act_layer,
                        nn.Dropout3d(0.1),
                        nn.BatchNorm3d(2*inter_num_ch))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(3*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.BatchNorm3d(4*inter_num_ch))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(6*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=(1,0,0), stride=2),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.BatchNorm3d(4*inter_num_ch),
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=2))

        self.conv2to3 = nn.MaxPool3d(2)
        self.conv3to4 = nn.MaxPool3d(2)

        self.init_model()

    def init_model(self):
        #pdb.set_trace()
        for block in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for layer in block.children():
                #print(layer)
                if isinstance(layer, nn.Conv3d):
                    for name, weight in layer.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_normal_(weight)
                        if 'bias' in name:
                            nn.init.constant_(weight, 0.0)

    def forward(self, x):
        #pdb.set_trace()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2to3 = self.conv2to3(conv1)
        conv3_input = torch.cat([conv2, conv2to3], dim=1)
        conv3 = self.conv3(conv3_input)
        conv3to4 = self.conv3to4(conv2)
        conv4_input = torch.cat([conv3, conv3to4], dim=1)
        conv4 = self.conv4(conv4_input)
        return conv4
'''

'''
class FeatureExtractor2(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu', arch='resnet'):
        super(FeatureExtractor2, self).__init__()

        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1, stride=2),
                        conv_act_layer,
                        nn.BatchNorm3d(inter_num_ch))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        conv_act_layer,
                        nn.Dropout3d(0.1),
                        nn.BatchNorm3d(2*inter_num_ch))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.BatchNorm3d(4*inter_num_ch))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=0, stride=2),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.BatchNorm3d(4*inter_num_ch))

        self.init_model()

    def init_model(self):
        #pdb.set_trace()
        for block in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for layer in block.children():
                #print(layer)
                if isinstance(layer, nn.Conv3d):
                    for name, weight in layer.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_normal_(weight)
                        if 'bias' in name:
                            nn.init.constant_(weight, 0.0)

    def forward(self, x):
        #pdb.set_trace()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4
'''

'''
class FeatureExtractor_ResNet2(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu', arch='resnet'):
        super(FeatureExtractor_ResNet2, self).__init__()

        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if arch == 'resnet':
            num_ch_list = [inter_num_ch, 2*inter_num_ch, 2*inter_num_ch, 4*inter_num_ch, 4*inter_num_ch]
        elif arch == 'resnet_small':
            num_ch_list = [16, 16, 24, 36, 36]

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, num_ch_list[0], kernel_size=3, padding=1),
                        nn.BatchNorm3d(num_ch_list[0]),
                        conv_act_layer)

        self.conv2 = nn.Sequential(
                        nn.Conv3d(num_ch_list[0], num_ch_list[1], kernel_size=3, stride=2, padding=1),
                        nn.Dropout3d(0.1),
                        nn.BatchNorm3d(num_ch_list[1]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[1], num_ch_list[1], kernel_size=3, padding=1))

        self.conv3 = nn.Sequential(
                        nn.BatchNorm3d(num_ch_list[0]+num_ch_list[1]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[0]+num_ch_list[1], num_ch_list[2], kernel_size=3, stride=2, padding=1),
                        nn.Dropout3d(0.1),
                        nn.BatchNorm3d(num_ch_list[2]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[2], num_ch_list[2], kernel_size=3, padding=1))

        self.conv4 = nn.Sequential(
                        nn.BatchNorm3d(num_ch_list[1]+num_ch_list[2]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[1]+num_ch_list[2], num_ch_list[3], kernel_size=3, stride=2, padding=1),
                        nn.Dropout3d(0.2),
                        nn.BatchNorm3d(num_ch_list[3]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[3], num_ch_list[3], kernel_size=3, padding=1))

        self.conv5 = nn.Sequential(
                        nn.BatchNorm3d(num_ch_list[2]+num_ch_list[3]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[2]+num_ch_list[3], num_ch_list[4], kernel_size=3, stride=2, padding=(1,0,0)),
                        nn.Dropout3d(0.2),
                        nn.BatchNorm3d(num_ch_list[4]),
                        conv_act_layer,
                        nn.Conv3d(num_ch_list[4], num_ch_list[4], kernel_size=2))

        self.conv2to3 = nn.MaxPool3d(2)
        self.conv3to4 = nn.MaxPool3d(2)
        self.conv4to5 = nn.MaxPool3d(2)

        self.init_model()

    def init_model(self):
        #pdb.set_trace()
        for block in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            for layer in block.children():
                #print(layer)
                if isinstance(layer, nn.Conv3d):
                    for name, weight in layer.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_normal_(weight)
                        if 'bias' in name:
                            nn.init.constant_(weight, 0.0)

    def forward(self, x):
        #pdb.set_trace()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2to3 = self.conv2to3(conv1)
        conv3_input = torch.cat([conv2, conv2to3], dim=1)
        conv3 = self.conv3(conv3_input)
        conv3to4 = self.conv3to4(conv2)
        conv4_input = torch.cat([conv3, conv3to4], dim=1)
        conv4 = self.conv4(conv4_input)
        conv4to5 = self.conv4to5(conv3)
        conv5_input = torch.cat([conv4, conv4to5], dim=1)
        conv5 = self.conv5(conv5_input)
        return conv5
'''

class FeatureExtractor_Ehsan(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu', arch='resnet'):
        super(FeatureExtractor_Ehsan, self).__init__()

        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(inter_num_ch),
                        conv_act_layer,
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(inter_num_ch),
                        conv_act_layer,
                        nn.MaxPool3d(2))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(2*inter_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(0.1),
                        nn.Conv3d(2*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(2*inter_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(0.1),
                        nn.MaxPool3d(2))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(4*inter_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(4*inter_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.MaxPool3d(2))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=0), #old pad 1
                        nn.BatchNorm3d(4*inter_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=0),
                        nn.BatchNorm3d(4*inter_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(0.2),
                        nn.MaxPool3d(2))

        self.init_model()

    def init_model(self):
        #pdb.set_trace()
        for block in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for layer in block.children():
                #print(layer)
                if isinstance(layer, nn.Conv3d):
                    for name, weight in layer.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_normal_(weight)
                        if 'bias' in name:
                            nn.init.constant_(weight, 0.0)

    def forward(self, x):
        # pdb.set_trace()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4

class SingleTimestep3DCNN(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, fe_arch='baseline'):
        super(SingleTimestep3DCNN, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8    # old 9

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, 2*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc3 = nn.Linear(2*fc_num_ch, num_output)

        self.init_model()

    def init_model(self):
        #pdb.set_trace()
        for layer in [self.fc1, self.fc2, self.fc3]:
            for name, weight in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(weight)
                if 'bias' in name:
                    nn.init.constant_(weight, 0.0)



    def forward(self, x, mask):
        conv4 = self.feature_extractor(x)
        conv4_flatten = conv4.view(x.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1)
        output = self.fc3(fc2)
        if self.num_cls == 0:
            output = F.relu(output)
        return [output]

class MultipleTimestepConcat(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5):
        super(MultipleTimestepConcat, self).__init__()

        self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2:
            num_output = 1
        else:
            num_output = num_cls

        num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        self.fc3 = nn.Linear(num_timestep*inter_num_ch, num_output)

    def forward(self, x, mask):
        bs, ts, img_x, img_y, img_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # x = x.view(bs*ts, img_x, img_y, img_z)
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        fc2_concat = fc2.view(bs, ts, -1).view(bs, -1)  # (bs, ts*16)
        output = self.fc3(fc2_concat)
        return [output]

class MultipleTimestepConcatMultipleOutput(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, fe_arch='baseline'):
        super(MultipleTimestepConcatMultipleOutput, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2:
            num_output = 1
        else:
            num_output = num_cls

        self.fc1 = nn.Sequential(
                        # nn.Linear(num_feat, 2*fc_num_ch),
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        # nn.Linear(2*fc_num_ch, fc_num_ch),
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        # self.fc3 = nn.Sequential(
        #                 nn.Linear(num_timestep*fc_num_ch, fc_num_ch),
        #                 fc_act_layer,
        #                 nn.Dropout3d(0.1),
        #                 nn.Linear(fc_num_ch, num_output))
        self.fc3 = nn.Linear(num_timestep*fc_num_ch, num_output)

        self.fc4 = nn.Linear(fc_num_ch, num_output)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts, img_x, img_y, img_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # x = x.view(bs*ts, img_x, img_y, img_z)
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        fc2_concat = fc2.view(bs, ts, -1).view(bs, -1)  # (bs, ts*16)
        output = self.fc3(fc2_concat)   # fused output
        fc4 = self.fc4(fc2) # single time output
        output_2 = fc4.view(bs, ts, -1)
        return [output, output_2]

class MultipleTimestepConcatMultipleOutputAvgPool(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, fe_arch='baseline'):
        super(MultipleTimestepConcatMultipleOutputAvgPool, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2:
            num_output = 1
        else:
            num_output = num_cls

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2_pool = nn.Sequential(
                        nn.Linear(2*fc_num_ch, fc_num_ch),
                        fc_act_layer)

        self.fc3 = nn.Linear(num_timestep*fc_num_ch, num_output)

        self.fc4 = nn.Linear(fc_num_ch, num_output)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts, img_x, img_y, img_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # x = x.view(bs*ts, img_x, img_y, img_z)
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        fc2_concat = fc2.view(bs, ts, -1).view(bs, -1)  # (bs, ts*16)
        output = self.fc3(fc2_concat)   # fused output

        fc2_reshape = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        fc2_avgpool = (fc2_reshape * mask.unsqueeze(-1)).mean(1).unsqueeze(1).repeat(1,ts,1)
        fc2_concat = torch.cat([fc2_reshape, fc2_avgpool], dim=-1)
        fc2_concat = self.fc2_pool(fc2_concat)
        fc2_concat = fc2_concat.view(bs, ts, -1)  # (bs, ts, 16)

        output_2 = self.fc4(fc2_concat) # single time output
        return [output, output_2]


class MultipleTimestepLSTM(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=False, init_lstm=False, rnn_type='LSTM', fe_arch='baseline'):
        super(MultipleTimestepLSTM, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls

        self.skip_missing = skip_missing

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3 = nn.Linear(lstm_num_ch, num_output)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        fc2_concat = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        #pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()
            # TODO: if skipping middle ts, need change
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        #lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        #fc3 = self.fc3(lstm_reshape)
        #output = fc3.view(bs, ts, -1)
        output = self.fc3(lstm)
        if self.num_cls == 0:
            output = F.relu(output)
        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat]
        else:
            return [output[:,-1,:], output]

class MultipleTimestepLSTMAvgPool(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=False, init_lstm=False, rnn_type='LSTM', fe_arch='baseline'):
        super(MultipleTimestepLSTMAvgPool, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls

        self.skip_missing = skip_missing

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        self.fc2_pool = nn.Sequential(
                        nn.Linear(2*fc_num_ch, fc_num_ch),
                        fc_act_layer)

        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3 = nn.Linear(lstm_num_ch, num_output)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        #pdb.set_trace()
        fc2_reshape = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        fc2_avgpool = (fc2_reshape * mask.unsqueeze(-1)).mean(1).unsqueeze(1).repeat(1,ts,1)
        fc2_concat = torch.cat([fc2_reshape, fc2_avgpool], dim=-1)
        fc2_concat = self.fc2_pool(fc2_concat)
        fc2_concat = fc2_concat.view(bs, ts, -1)  # (bs, ts, 16)
        #pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()
            # TODO: if skipping middle ts, need change
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        #lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        #fc3 = self.fc3(lstm_reshape)
        #output = fc3.view(bs, ts, -1)
        output = self.fc3(lstm)
        if self.num_cls == 0:
            output = F.tanh(output)  # need to check with dataset
        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat, fc2_reshape]
        else:
            return [output[:,-1,:], output]

class LongitudinalPoolingLayer(nn.Module):
    def __init__(self, num_timestep=5):
        super(LongitudinalPoolingLayer, self).__init__()
        self.num_timestep = num_timestep

    def forward(self, feat):
        feat_lp = torch.zeros_like(feat)
        #pdb.set_trace()
        for idx in range(self.num_timestep):
            if idx == self.num_timestep - 1:
                feat_lp[:, idx, :] = feat[:, idx, :]
            else:
                feat_lp[:, idx, :] = feat[:, idx+1:, :].sum(dim=1)
        return feat_lp


class MultipleTimestepLSTMFutureAvgPool(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=True, init_lstm=True, rnn_type='GRU', fe_arch='ehsan'):
        super(MultipleTimestepLSTMFutureAvgPool, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls

        self.skip_missing = skip_missing

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        self.lp_layer = LongitudinalPoolingLayer(num_timestep)

        self.fc2_pool = nn.Sequential(
                        nn.Linear(2*fc_num_ch, fc_num_ch),
                        fc_act_layer)

        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3 = nn.Linear(lstm_num_ch, num_output)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        #pdb.set_trace()
        fc2_reshape = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        fc2_avgpool = self.lp_layer(fc2_reshape)
        #fc2_avgpool = (fc2_reshape * mask.unsqueeze(-1)).mean(1).unsqueeze(1).repeat(1,ts,1)
        fc2_concat = torch.cat([fc2_reshape, fc2_avgpool], dim=-1)
        fc2_concat = self.fc2_pool(fc2_concat)
        fc2_concat = fc2_concat.view(bs, ts, -1)  # (bs, ts, 16)
        #pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()
            # TODO: if skipping middle ts, need change
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        #lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        #fc3 = self.fc3(lstm_reshape)
        #output = fc3.view(bs, ts, -1)
        output = self.fc3(lstm)
        if self.num_cls == 0:
            output = F.tanh(output)  # need to check with dataset
        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat, fc2_reshape]
        else:
            return [output[:,-1,:], output]




class MultipleTimestepLSTMAvgPoolOrdinary(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=False, init_lstm=False, rnn_type='LSTM', fe_arch='baseline'):
        super(MultipleTimestepLSTMAvgPoolOrdinary, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls <= 2:
            raise ValueError('This Model only support multi-class classification')

        self.num_cls = num_cls

        self.skip_missing = skip_missing

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        self.fc2_pool = nn.Sequential(
                        nn.Linear(2*fc_num_ch, fc_num_ch),
                        fc_act_layer)

        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3_1 = nn.Linear(lstm_num_ch, 1)
        self.fc3_2 = nn.Linear(lstm_num_ch, 1)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        #pdb.set_trace()
        fc2_reshape = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        fc2_avgpool = (fc2_reshape * mask.unsqueeze(-1)).mean(1).unsqueeze(1).repeat(1,ts,1)
        fc2_concat = torch.cat([fc2_reshape, fc2_avgpool], dim=-1)
        fc2_concat = self.fc2_pool(fc2_concat)
        fc2_concat = fc2_concat.view(bs, ts, -1)  # (bs, ts, 16)
        #pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()
            # TODO: if skipping middle ts, need change
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        #lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        #fc3 = self.fc3(lstm_reshape)
        #output = fc3.view(bs, ts, -1)
        output_1 = self.fc3_1(lstm)
        output_2 = self.fc3_2(lstm)
        output = torch.cat([output_1, output_2], -1)

        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat]
        else:
            return [output[:,-1,:], output]



class MultipleTimestepLSTMAvgPoolDate(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=False, init_lstm=False, rnn_type='LSTM', fe_arch='baseline'):
        super(MultipleTimestepLSTMAvgPoolDate, self).__init__()

        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls

        self.skip_missing = skip_missing

        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(4*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))
        self.fc2_date = nn.Sequential(
                        nn.Linear(fc_num_ch, fc_num_ch),
                        fc_act_layer,
                        nn.Linear(fc_num_ch, 1),
                        nn.ReLU())

        self.fc2_pool = nn.Sequential(
                        nn.Linear(2*fc_num_ch, fc_num_ch),
                        fc_act_layer)

        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3 = nn.Linear(lstm_num_ch, num_output)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(x)   # (bs*ts,16,2,4,4)
        conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1) # (bs*ts,16)
        #pdb.set_trace()
        fc2_reshape = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        # date interval
        interval = self.fc2_date(fc2_reshape)
        # pooling
        fc2_avgpool = (fc2_reshape * mask.unsqueeze(-1)).mean(1).unsqueeze(1).repeat(1,ts,1)
        fc2_concat = torch.cat([fc2_reshape, fc2_avgpool], dim=-1)
        fc2_concat = self.fc2_pool(fc2_concat)
        fc2_concat = fc2_concat.view(bs, ts, -1)  # (bs, ts, 16)
        #pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()
            # TODO: if skipping middle ts, need change
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        #lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        #fc3 = self.fc3(lstm_reshape)
        #output = fc3.view(bs, ts, -1)
        output = self.fc3(lstm)
        if self.num_cls == 0:
            output = F.relu(output)
        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat, interval]
        else:
            return [output[:,-1,:], output]



class MultipleTimestepConcatMetadata(nn.Module):
    def __init__(self, in_num_ch=1, meta_size=20, fc_num_ch=16,
                fc_act='tanh', num_cls=2, num_timestep=5):
        super(MultipleTimestepConcatMetadata, self).__init__()

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2:
            num_output = 1
        else:
            num_output = num_cls

        self.fc1 = nn.Sequential(
                        nn.Linear(5*meta_size, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))
        # self.fc2 = nn.Sequential(
        #                 nn.Linear(4*fc_num_ch, fc_num_ch),
        #                 fc_act_layer)

        self.fc3 = nn.Linear(fc_num_ch, num_output)


    def forward(self, x, mask):
        bs, ts = x.shape[0], x.shape[1]
        x = x.view(bs, -1)
        # x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,20) -> (bs*ts,20)
        fc1 = self.fc1(x)
        # fc2 = self.fc2(fc1)
        # fc2_concat = fc2.view(bs, ts, -1).view(bs, -1)
        fc3 = self.fc3(fc1)
        return [fc3]


class MultipleTimestepLSTMMetadata(nn.Module):
    def __init__(self, in_num_ch=1, meta_size=20, fc_num_ch=16, lstm_num_ch=16,
                fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=False, init_lstm=False):
        super(MultipleTimestepLSTMMetadata, self).__init__()

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2:
            num_output = 1
        else:
            num_output = num_cls

        self.skip_missing = skip_missing

        self.fc1 = nn.Sequential(
                        nn.Linear(meta_size, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))
        # self.fc2 = nn.Sequential(
        #                 nn.Linear(4*fc_num_ch, fc_num_ch),
        #                 fc_act_layer)

        self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)

        self.fc3 = nn.Linear(lstm_num_ch, num_output)
        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,20) -> (bs*ts,20)
        fc1 = self.fc1(x)
        # fc2 = self.fc2(fc1)
        # fc2_concat = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        fc2_concat = fc1.view(bs, ts, -1)
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        fc3 = self.fc3(lstm_reshape)
        output = fc3.view(bs, ts, -1)
        if self.skip_missing:
            output_last = torch.cat([output[i, num_ts_list[i].long()-1, :] for i in range(bs)], dim=0).unsqueeze(-1)
            return [output_last, output]
        else:
            return [output[:,-1,:], output]


class MultipleTimestepLSTMMultimodal(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), meta_size=20, inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3,
                conv_act='relu', fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=False, init_lstm=False):
        super(MultipleTimestepLSTMMultimodal, self).__init__()

        self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)

        if num_cls == 2:
            num_output = 1
        else:
            num_output = num_cls

        self.skip_missing = skip_missing

        # num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        # self.fc1 = nn.Sequential(
        #                 nn.Linear(num_feat, 4*fc_num_ch),
        #                 fc_act_layer,
        #                 nn.Dropout3d(0.1))
        self.fc1 = nn.Sequential(
                        nn.Linear(inter_num_ch, 4*fc_num_ch),
                        fc_act_layer,
                        nn.Dropout3d(0.1))

        self.fc1_meta = nn.Sequential(
                        nn.Linear(meta_size, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        self.fc2 = nn.Sequential(
                        nn.Linear(5*fc_num_ch, fc_num_ch),
                        fc_act_layer)
                        # nn.Dropout3d(0.1))

        self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)

        self.fc3 = nn.Linear(lstm_num_ch, num_output)
        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, img, meta, mask):
        bs, ts = img.shape[0], img.shape[1]
        img = torch.cat([img[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        img = img.unsqueeze(1)  # (bs*ts,1,32,64,64)
        conv4 = self.feature_extractor(img)   # (bs*ts,16,2,4,4)
        # conv4_flatten = conv4.view(conv4.shape[0], -1)
        # pdb.set_trace()
        conv4_flatten = F.avg_pool3d(conv4, kernel_size=(2,4,4))
        fc1 = self.fc1(conv4_flatten.squeeze())   # (bs*ts, 64)

        meta = torch.cat([meta[b,...] for b in range(bs)], dim=0)  # (bs,ts,20) -> (bs*ts,20)
        fc1_meta = self.fc1_meta(meta)  # (bs*ts, 16)

        fc1_concat = torch.cat([fc1, fc1_meta], dim=1)  # (bs*ts, 64+16)
        fc2 = self.fc2(fc1_concat) # (bs*ts,16)
        fc2_concat = fc2.view(bs, ts, -1)  # (bs, ts, 16)
        # pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        fc3 = self.fc3(lstm_reshape)
        output = fc3.view(bs, ts, -1)
        if self.skip_missing:
            output_last = torch.cat([output[i, num_ts_list[i].long()-1, :] for i in range(bs)], dim=0).unsqueeze(-1)
            return [output_last, output]
        else:
            return [output[:,-1,:], output]
