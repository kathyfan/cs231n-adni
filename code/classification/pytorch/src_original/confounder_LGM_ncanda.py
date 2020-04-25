import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
import scipy.io
import pdb

fold = 1
data_dict = np.load('feature_ncanda_fold'+str(fold)+'_test.npy').item()
pdb.set_trace()
subj_id_test = data_dict['subj_id']
gender_test = 2*data_dict['gender'] - 1 # M: 1, F: -1
#gender = 1+data_dict['gender']
label_ts_test = data_dict['label_ts']
mask_test = data_dict['mask'].astype(np.bool)
pred_ts_test = data_dict['pred_ts']
fc2_reshape_test = data_dict['fc2_reshape']
fc2_concat_test = data_dict['fc2_concat']

data_dict = np.load('feature_ncanda_fold'+str(fold)+'_train.npy').item()
pdb.set_trace()
subj_id = data_dict['subj_id']
gender = 2*data_dict['gender'] - 1 # M: 1, F: -1
#gender = 1+data_dict['gender']
label_ts = data_dict['label_ts']
mask = data_dict['mask'].astype(np.bool)
pred_ts = data_dict['pred_ts']
fc2_reshape = data_dict['fc2_reshape']
fc2_concat = data_dict['fc2_concat']

pdb.set_trace()
#with open('ncanda_subj_id.txt') as file:
#    subj_list = file.readlines()
ses_dict = np.load('../data/NCANDA/ncanda_ses.npy').item()
ses_list = []
for i, s_id in enumerate(subj_id):
    ses = ses_dict[s_id]
    if np.isnan(ses):
        ses = 0
    ses_list.append(ses)
ses_list = np.array(ses_list)
#ses_list /= ses_list.max()

ses_list_test = []
for i, s_id in enumerate(subj_id_test):
    ses = ses_dict[s_id]
    if np.isnan(ses):
        ses = 0
    ses_list_test.append(ses)
ses_list_test = np.array(ses_list_test)
#ses_list_test /= ses_list_test.max()
ses_max = max(ses_list.max(), ses_list_test.max())
ses_list /= ses_max
ses_list_test /= ses_max

gender_rep = np.tile(gender.reshape(-1,1), (1,5))
gender_sel = gender_rep[mask].reshape(-1,1)
ses_rep = np.tile(ses_list.reshape(-1,1), (1,5))
ses_sel = ses_rep[mask].reshape(-1,1)
label_sel = label_ts[mask]
pred_sel = pred_ts[mask]
fc2_reshape_sel = fc2_reshape[mask]
fc2_concat_sel = fc2_concat[mask]

gender_rep_test = np.tile(gender_test.reshape(-1,1), (1,5))
gender_sel_test = gender_rep_test[mask_test].reshape(-1,1)
ses_rep_test = np.tile(ses_list_test.reshape(-1,1), (1,5))
ses_sel_test = ses_rep_test[mask_test].reshape(-1,1)
label_sel_test = label_ts_test[mask_test]
pred_sel_test = pred_ts_test[mask_test]
fc2_reshape_sel_test = fc2_reshape_test[mask_test]
fc2_concat_sel_test = fc2_concat_test[mask_test]


pdb.set_trace()

num_feat = fc2_concat_sel.shape[1]
residual_all = []
residual_all_test = []
for idx in range(num_feat):
    #pdb.set_trace()
    feat = fc2_concat_sel[:, idx]
    input_conf = np.concatenate([ses_sel, gender_sel], axis=1)
    input_conf_test = np.concatenate([ses_sel_test, gender_sel_test], axis=1)

    reg = LinearRegression().fit(input_conf, feat)
    pred_hat = reg.predict(input_conf)
    residual = feat - pred_hat
    residual_all.append(residual.reshape(-1,1))

    feat_test = fc2_concat_sel_test[:, idx]
    pred_hat_test = reg.predict(input_conf_test)
    residual_test = feat_test - pred_hat_test
    residual_all_test.append(residual_test.reshape(-1,1))

    #beta0 = reg.predict(np.array([[0,0]]))
    #beta1 = reg.predict(np.array([[1,0]])) - beta0
    #beta2 = reg.predict(np.array([[0,1]])) - beta0
    #print(p_value_g, p_value_s, beta0, beta1, beta2)
residual_all = np.concatenate(residual_all, axis=1)
residual_all_test = np.concatenate(residual_all_test, axis=1)
pdb.set_trace()

data_dict_new = {'gender':gender_sel, 'ses':ses_sel, 'label':label_sel, 'pred':pred_sel, 'fc2_reshape':fc2_reshape_sel, 'fc2_concat':fc2_concat_sel, 'res':residual_all}
scipy.io.savemat('feature_ncanda_fold'+str(fold)+'_train.mat', data_dict_new)
np.save('feature_ncanda_fold'+str(fold)+'_sel_train.npy', data_dict_new)
data_dict_new_test = {'gender':gender_sel_test, 'ses':ses_sel_test, 'label':label_sel_test, 'pred':pred_sel_test, 'fc2_reshape':fc2_reshape_sel_test, 'fc2_concat':fc2_concat_sel_test, 'res':residual_all_test}
scipy.io.savemat('feature_ncanda_fold'+str(fold)+'_test.mat', data_dict_new_test)
np.save('feature_ncanda_fold'+str(fold)+'_sel_test.npy', data_dict_new_test)
'''
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(residual_all, label_sel)
print(clf.score(residual_all, label_sel))
print(clf.score(residual_all_test, label_sel_test))
pred1 = clf.predict(residual_all_test)
print(accuracy_score(label_sel_test, pred1))

pdb.set_trace()
clf2 = SVC(gamma='auto')
clf2.fit(fc2_concat_sel, label_sel)
print(clf2.score(fc2_concat_sel, label_sel))
print(clf2.score(fc2_concat_sel_test, label_sel_test))
pred2 = clf.predict(fc2_concat_test)
print(accuracy_score(label_sel_test, pred2))
'''
