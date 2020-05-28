import numpy as np
import nibabel as nib
import scipy as sp
import csv

# get ages of images corresponding to data in fold
def get_ages(fold=4):
    reader = csv.DictReader(open("../metadata/ADNI1AND2.csv"))

    # dedup csv based on subject_id. create a dictionary so that we don't
    # have to loop through redundant csv later.
    ids = set()
    items = {}
    for row in reader:
        if row['Subject_ID'] not in ids:
            items['Subject_ID'] = row['Age']
            ids.add(row['Subject_ID'])

    # load image names
    file_idx = np.genfromtxt('./subjects_idx.txt', dtype='str')
    # load fold indices
    fold_idx = np.loadtxt('fold.txt')
    want_idx = (fold_idx == fold)
    ages = np.zeros_like(want_idx) # empty list

    for i in want_idx:
        name_full = file_idx[i]
        name_short = name_full[:9] # first 10 characters

        # look in dictionary to match row with name_short and extract age
        for k, v in items:
            if k == name_short:
                ages[i] = v
    return ages

def get_test_data_unaugmented():
    # Load the data
    file_idx = np.genfromtxt('./subjects_idx.txt', dtype='str')
    fold_idx = np.loadtxt('fold.txt')                   # to keep same-patient images together
    label = np.loadtxt('dx.txt')
    np.random.seed(seed=0)

    subject_num = file_idx.shape[0]

    # Only use a patch from each input image
    patch_x = 64
    patch_y = 64
    patch_z = 64
    min_x = 0
    min_y = 0
    min_z = 0
    i = 0

    data = np.zeros((subject_num, patch_x, patch_y, patch_z,1))

    for img_idx in file_idx:
        filename_full = '/Users/elissali/Documents/GitHub/cs231n-adni/data/' + img_idx
        # '/fs/neurosci01/qingyuz/3dcnn/ADNI/img_64_longitudinal/'
        img = nib.load(filename_full)
        img_data = img.get_fdata()

        data[i,:,:,:,0] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z]
        data[i,:,:,:,0] = (data[i,:,:,:,0] - np.mean(data[i,:,:,:,0])) / np.std(data[i,:,:,:,0])
        i += 1

    test_idx = (fold_idx == 4)
    test_data = data[test_idx]
    test_label = label[test_idx]

    return test_data, test_label

test_data, test_label = get_test_data_unaugmented()
print(test_data.shape, test_label.shape)
np.save("test_data", test_data)
np.save("test_label", test_label)

def get_data():
    # Load the data
    file_idx = np.genfromtxt('./subjects_idx.txt', dtype='str') 
    fold_idx = np.loadtxt('fold.txt')                   # to keep same-patient images together
    label = np.loadtxt('dx.txt')
    np.random.seed(seed=0)

    subject_num = file_idx.shape[0]
    print("data.py line 13: ", subject_num)

    # Only use a patch from each input image
    patch_x = 64
    patch_y = 64
    patch_z = 64
    min_x = 0
    min_y = 0
    min_z = 0
    i = 0

    data = np.zeros((subject_num, patch_x, patch_y, patch_z,1))
    print("data.py: line 26")

    for img_idx in file_idx:
        print("data.py: line 29: ", i)
        filename_full = '/Users/elissali/Documents/GitHub/cs231n-adni/data/' + img_idx
        # '/fs/neurosci01/qingyuz/3dcnn/ADNI/img_64_longitudinal/'
        img = nib.load(filename_full)
        img_data = img.get_fdata()

        data[i,:,:,:,0] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data[i,:,:,:,0] = (data[i,:,:,:,0] - np.mean(data[i,:,:,:,0])) / np.std(data[i,:,:,:,0])
        i += 1

    print("data.py: line 39: finished loading stuff")

    # partition entire dataset into train, val, test
    # fold_idx goes from (0,4); separating these arbitrarily based on fold_idx
    train_idx = (fold_idx <= 2) 
    val_idx = (fold_idx == 3)
    test_idx = (fold_idx == 4)

    train_data = data[train_idx]
    train_label = label[train_idx]

    val_data = data[val_idx]
    val_label = label[val_idx]

    test_data = data[test_idx]
    test_label = label[test_idx]

    print("data.py: line 56: finished partitioning")

    # Augment data
    augment_size = 1024
    augment_size_val = 256
    augment_size_test = 256
    
    train_data_pos = train_data[train_label==1]
    train_data_neg = train_data[train_label==0]
    train_data_pos_aug = augment_by_transformation(train_data_pos, augment_size)
    train_data_neg_aug = augment_by_transformation(train_data_neg, augment_size)
    train_data_aug = np.concatenate((train_data_neg_aug, train_data_pos_aug), axis=0)
    print("data.py: line 68")
    train_label_aug = np.zeros((augment_size * 2,))
    train_label_aug[augment_size:] = 1

    val_data_pos = val_data[val_label==1]
    val_data_neg = val_data[val_label==0]
    val_data_pos_aug = augment_by_transformation(val_data_pos, augment_size_val)
    val_data_neg_aug = augment_by_transformation(val_data_neg, augment_size_val)
    val_data_aug = np.concatenate((val_data_neg_aug, val_data_pos_aug), axis=0)
    print("data.py: line 77")
    val_label_aug = np.zeros((augment_size_val * 2,))
    val_label_aug[augment_size_val:] = 1

    test_data_pos = test_data[test_label==1]
    test_data_neg = test_data[test_label==0]
    test_data_pos_aug = augment_by_transformation(test_data_pos, augment_size_test)
    test_data_neg_aug = augment_by_transformation(test_data_neg, augment_size_test)
    test_data_aug = np.concatenate((test_data_neg_aug, test_data_pos_aug), axis=0)

    test_label_aug = np.zeros((augment_size_test * 2,))
    test_label_aug[augment_size_test:] = 1

    print("data.py: line 90: done augmenting; saving np arrays")
    np.save("train_aug", train_data_aug)
    np.save("val_aug", val_data_aug)
    np.save("test_aug", test_data_aug)
    np.save("train_label_aug", train_label_aug)
    np.save("val_label_aug", val_label_aug)
    np.save("test_label_aug", test_label_aug)
    return train_data_aug, val_data_aug, test_data_aug, train_label_aug, val_label_aug, test_label_aug

## Data Augmentation
def augment_by_transformation(data,n):
    print("data.py line 98: inside augment_by_transformation: ", data)
    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]           # number of examples we actually have
        m = n - raw_n                   # m = number of examples to generate to get n total examples (n = augment_size)
        new_data = np.zeros((m,data.shape[1],data.shape[2],data.shape[3],1))
        for i in range(0,m):
            idx = np.random.randint(0,raw_n)
            new_data[i] = data[idx].copy()
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(1,0),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(0,2),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5),axes=(1,2),reshape=False)
            new_data[i,:,:,:,0] = sp.ndimage.shift(new_data[i,:,:,:,0],np.random.uniform(-0.5,0.5))
            print("data.py line 112: inside augment_by_transformation: ", i)
        data = np.concatenate((data, new_data), axis=0)
        return data

# print(get_data('subjects_idx.txt')) #### In case you want to check if the get_data function works properly