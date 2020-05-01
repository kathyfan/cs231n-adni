import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_num_ch=1, inter_num_ch=16, conv_act='relu'):
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

class SingleTimestep3DCNN(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16,
                conv_act='relu', fc_act='tanh'):
        super(SingleTimestep3DCNN, self).__init__()

        self.feature_extractor = FeatureExtractor(in_num_ch, inter_num_ch,conv_act)
        num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))

        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        else:
            raise ValueError('No implementation of ', fc_act)

        num_output = 1
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
