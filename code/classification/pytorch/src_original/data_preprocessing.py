import os
import numpy as np
import pandas as pd
import pdb

csv_path = '../data/NCANDA_Upto_Year5-20-01-06.csv'
# meta_name = ['ysr_external_raw', 'ses_parent_base', 'np_wais4_rawscore', 'lssaga3_youth_as14b', 'cnp_pvrt_pvrt_eff',
#             'cnp_pmat24a_pmat24_a_cr', 'dd1000_logk_6mo', 'lssaga3_youth_as2b', 'lssaga3_youth_as9', 'lssaga3_youth_as11',
#             'lssaga3_youth_as2a', 'np_reyo_copy_raw', 'lssaga3_youth_as6b', 'lssaga3_youth_as15', 'lssaga3_youth_as20',
#             'cnp_spcptnl_scpt_tprt', 'ysr_internal_raw', 'lssaga3_youth_as19', 'lssaga3_youth_as14', 'lssaga3_youth_as16']
meta_name = ['ysr_external_raw', 'ses_parent_base', 'np_wais4_rawscore', 'cnp_pvrt_pvrt_eff',
            'cnp_pmat24a_pmat24_a_cr', 'dd1000_logk_6mo', 'np_reyo_copy_raw', 'cnp_spcptnl_scpt_tprt', 'ysr_internal_raw']
data = pd.read_csv(csv_path, usecols=['subject', 'sex', 'visit', 'DrkClass', 'DrkClassMax', 'visit_age',]+meta_name)

# preprocessed timestep and label
if False:
    subj_data = {}
    for idx, row in data.iterrows():
        subj_id = row['subject']
        visit_id = row['visit']
        label = row['DrkClassMax']
        # pdb.set_trace()
        # metadata = row[-1-len(meta_name):-1]
        sex = 1 if row['sex']=='M' else 0
        if subj_id in subj_data:
            # not first timestep
            missing_timestep = int(visit_id.split('_')[-1][0]) - subj_data[subj_id]['last_visit'] - 1
            for i in range(missing_timestep):
                subj_data[subj_id]['visit'].append('none')
            subj_data[subj_id]['visit'].append(visit_id)
            subj_data[subj_id]['last_visit'] = int(visit_id.split('_')[-1][0])
        else:
            # first timestep
            # pdb.set_trace()
            subj_data[subj_id] = {'visit': [visit_id], 'label': label, 'sex':sex}
            if visit_id == 'baseline':
                subj_data[subj_id]['first_visit'] = 0
            else:
                subj_data[subj_id]['first_visit'] = int(visit_id.split('_')[-1][0])
            subj_data[subj_id]['last_visit'] = subj_data[subj_id]['first_visit']

    max_timestep = 0
    for subj_id, info in subj_data.items():
        num_timestep = len(info['visit'])
        max_timestep = max(max_timestep, num_timestep)
    print('Number of subjects: ', len(subj_data))
    print('Max number of timesteps: ', max_timestep)

    subj_info_all = []
    for subj_id, info in subj_data.items():
        # info['label'] = info['sex']
        subj_info = [subj_id, info['label']]
        subj_info.extend(info['visit'])
        for i in range(max_timestep-len(info['visit'])):
            subj_info.append('none')
        subj_info_all.append(subj_info)
    df = pd.DataFrame(subj_info_all,
                    columns=['subj_id', 'label', 'timestep_1', 'timestep_2', 'timestep_3', 'timestep_4', 'timestep_5', 'timestep_6'])
    df.to_csv('../data/NCANDA_preprocessed.csv')

# preprocess metadata
if True:
    df = pd.read_csv('../data/NCANDA_preprocessed.csv', usecols=['subj_id', 'label', 'timestep_1', 'timestep_2', 'timestep_3', 'timestep_4', 'timestep_5'])
    subj_metadata = {}
    metadata_mean = data.loc[:,meta_name].mean(axis=0, skipna=True)
    metadata_std = data.loc[:,meta_name].std(axis=0, skipna=True)
    metadata_zscore = (data.loc[:,meta_name] - metadata_mean) / metadata_std
    metadata_zscore.fillna(0, inplace=True)
    for idx, row in df.iterrows():
        subj_id = row['subj_id']
        visit_id_list = row[2:]
        metadata = np.zeros((len(visit_id_list), len(meta_name)))
        for i, visit_id in enumerate(visit_id_list):
            data_row = metadata_zscore.loc[(data['subject']==subj_id) & (data['visit']==visit_id)]
            if data_row.shape[0] != 0:
                metadata[i] = data_row.loc[:,meta_name]
        subj_metadata[subj_id] = metadata
    subj_metadata['metadata_param'] = len(meta_name)
    np.save('../data/NCANDA_metadata_preprocessed_small.npy', subj_metadata)
