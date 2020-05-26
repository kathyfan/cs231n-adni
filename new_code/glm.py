import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from data import *


img_size = 4 # output of last maxpool3d layer: images of (4, 4, 4)
n_filters = 16 # 16 filters in last conv layer
n_feat = img_size*img_size*img_size*n_filters # number of features after flattening
n_samples = 100 # number of subjects
labels = None # set of labels (100,) -> use saved val_labels np array
features = None # (n_samples, n_feat)
cf = get_ages() # (n_samples,) -> ages

# y = c + a * mf + b * cf
# seems like c = constant, mf = labels, cf = confounding variable
# feature i is confounding if p[:,1] < 0.05 for feature i
def glm():
    p = np.zeros((n_feat, 3))

    for i in range(n_feat):
        X = np.zeros((n_samples, 3))
        X[:,0] = labels
        X[:, 1] = cf
        X[:, 2] = 1 # constant term?

        """
        first input: endog, 1d or 2d. I guess here just 1d. 
        second input: exog, (nobs, k). nobs=n_samples is number of observations and k=3 is number of regressors
        """
        glm_model = sm.GLM(features[:,i], X)
        glm_results = glm_model.fit()
        p[i,:] = glm_results.pvalues

    # calculate mask based on p values
    mask = (p[:,1]<0.05)
    plt.imshow([mask])
    plt.title("Feature mask (yellow = confounded by cf)")