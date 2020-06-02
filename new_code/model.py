import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_num_ch=1, inter_num_ch=16, conv_act='relu'):
        super(FeatureExtractor, self).__init__()

        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(inter_num_ch),
                        nn.MaxPool3d(2))
        # output of conv1: images scaled to (32, 32, 32)

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(2*inter_num_ch),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(0.1))
        # output of conv2: images scaled to (16, 16, 16)

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(4*inter_num_ch),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(0.1))
        # output of conv3: images scaled to (8, 8, 8)

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        conv_act_layer,
                        nn.BatchNorm3d(inter_num_ch),
                        nn.MaxPool3d(2))
        # output of conv4: images scaled to (4, 4, 4)

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
        self.num_cls = 2

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

    def forward(self, x):
        conv4 = self.feature_extractor(x)
        conv4_flatten = conv4.view(x.shape[0], -1)
        fc1 = self.fc1(conv4_flatten)
        fc2 = self.fc2(fc1)
        output = self.fc3(fc2)
        return output

def test_SingleTimestep3DCNN():
    x = torch.zeros((32, 1, 64, 64, 64)) # minibatch size 32, image size (1, 64, 64, 64)
    model = SingleTimestep3DCNN(in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16, fc_num_ch=16,
                                conv_act='relu', fc_act='tanh')
    scores = model(x)
    print(scores.size()) # should be (32, 2)

#test_SingleTimestep3DCNN()

""" confounder-aware model classes below"""
class BinaryMask(nn.Module):

    def __init__(self, output_dim, mask, **kwargs):
        self.output_dim = output_dim
        self.mask = torch.Tensor(mask)
        
        super(BinaryMask, self).__init__(**kwargs)

    def forward(self, x):
        return x * self.mask 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# trained checkpoint must exist and be in the expected path 
class Confounder3DCNN(nn.Module):
    def __init__(self, mask=mask, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, fc_num_ch=16,
                conv_act='relu', fc_act='tanh'):
        super(Confounder3DCNN, self).__init__()
        
        # load previous model
        net = SingleTimestep3DCNN(in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, fc_num_ch=16,
                                    conv_act='relu', fc_act='tanh').to(torch.device('cpu'))
        net.load_state_dict(torch.load('../ckpt/2020_5_18_16_42/epoch031.pth.tar')['model'])
        net.eval()

        self.feature_extractor = net.feature_extractor # use feature extractor from existing model

        num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))

        num_output = 1
        self.num_cls = 2
        fc_act_layer = nn.Tanh()
        
        self.binmask = model.BinaryMask(1024, mask)

        self.fc1 = net.fc1    # use fc1 from existing model
        self.fc2 = net.fc2    # use fc2 from existing model
        self.fc3 = net.fc3    # use fc3 from existing model

#         self.init_model()

    def init_model(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            for name, weight in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(weight)
                if 'bias' in name:
                    nn.init.constant_(weight, 0.0)

    def forward(self, x):
        conv4 = self.feature_extractor(x)
        conv4_flatten = conv4.view(x.shape[0], -1)
        conv4_mask = self.binmask(conv4_flatten)   # apply binary_mask
        fc1 = self.fc1(conv4_mask)
        print(fc1.shape)
        fc2 = self.fc2(fc1)
        output = self.fc3(fc2)
        return output