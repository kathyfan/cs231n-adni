import numpy as np
import torch
import statsmodels.api as sm
from matplotlib import pyplot as plt
import model, data

test_data = np.load("test_data.npy")
test_labels = np.load("test_label.npy")
img_size = 4 # output of last maxpool3d layer: images of (4, 4, 4)
n_filters = 16 # 16 filters in last conv layer
n_feat = img_size*img_size*img_size*n_filters # number of features after flattening
n_samples = 251 # number of subjects in unaugmented test set
labels = torch.Tensor(test_labels) # set of labels (n_samples,)
test_data = torch.Tensor(np.reshape(test_data, (251, 1, 64, 64, 64)))

def get_feats(data):
    print("starting get_feats")
    # load checkpoint of pre-trained model
    net = model.SingleTimestep3DCNN(in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, fc_num_ch=16,
                                    conv_act='relu', fc_act='tanh').to(torch.device('cpu'))
    net.load_state_dict(torch.load('../ckpt/2020_5_18_16_42/epoch031.pth.tar')['model'])

    # set to eval mode
    net.eval()

    feats = net.feature_extractor(data)
    print(feats.shape) # should be (251, 16, 4, 4, 4)
    flat_feats = feats.view(data.shape[0], -1) 
    print(flat_feats.shape) # should be (251, 1024)
    print("finish get_feats")
    
    flat_feats_np = flat_feats.cpu().detach().numpy()
    np.save("flat_feats", flat_feats_np)
    return flat_feats

# y = c + a * mf + b * cf
# seems like c = constant, mf = labels, cf = confounding variable
# feature i is confounding if p[:,1] < 0.05 for feature i
def perform_glm():
    p = np.zeros((n_feat, 3))
    
    cf = data.nb_get_ages()
    features = get_feats(test_data)
    feats = features.cpu().detach().numpy()

    num_error = 0
    feat_errored = []
    for i in range(n_feat):
        print("i: ", i)
        X = np.zeros((n_samples, 3))
        X[:,0] = labels
        X[:, 1] = cf
        X[:, 2] = 1 # constant term?

        """
        first input: endog, 1d or 2d. I guess here just 1d. 
        second input: exog, (nobs, k). nobs=n_samples is number of observations and k=3 is number of regressors
        """        
        glm_model = sm.GLM(feats[:,i], X)
        try:
            glm_results = glm_model.fit()
            p[i,:] = glm_results.pvalues
        except:
            p[i,:] = 1
            num_error += 1
            feat_errored.append(i)

    # calculate mask based on p values
    mask = (p[:,1]<0.05)
    num_masked = 0
    feat_masked = []
    for i in range(n_feat):
        if mask[i]:
            num_masked += 1
            feat_masked.append(i)
#     plt.imshow([mask[900:1000]])
#     plt.title("Feature mask (yellow = confounded by cf)")
    
    print("numbered of masked features: ", num_masked)
    print("list of masked features: i = ")
    print(feat_masked)
    print("numbered of errored features: ", num_error)
    print("list of errors: i = ")
    print(feat_errored)
    return mask

#perform_glm()