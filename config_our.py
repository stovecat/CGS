
import os
import sys
import json
import datetime
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd

from transformers import AdamW, get_linear_schedule_with_warmup
#from fairseq.models.bart import BARTModel
#from transformers import BartTokenizer

from config_m2m import *
from utils_en import *
from model_utils import FocalLoss, LDAMLoss
from model import BERTClassifier

#def parse_args():

#    return parser.parse_args()

#ARGS = parse_args()
print('== Arguments ==')
print(ARGS)

# legacy code - Needs to be refactored
set_global_variables(ARGS.gpu, ARGS.device, ARGS.MAX_LEN, ARGS.batch_size)

# Real = [ATIS, TREC], Imbalance manipulation = [SNIPS]
if ARGS.dataset == 'ATIS': 
    N_SAMPLES = -1
elif ARGS.dataset == 'SNIPS':
    N_SAMPLES = 1000
elif ARGS.dataset == 'TREC':
    N_SAMPLES = 1000
else:
    print('Dataset:', ARGS.dataset)
    raise NotImplementedError()

random.seed(ARGS.random_seed)
np.random.seed(ARGS.random_seed)
torch.manual_seed(ARGS.random_seed)
torch.cuda.manual_seed_all(ARGS.random_seed)

# To be refactored
model_name = '%s_%s_%s_classifier_%s_%s_%s_%s.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), ARGS.loss_type, ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel)

print('\n== Load data ==')

bert, tokenizer = get_bert()
labeled_train_data, labeled_valid_data, labeled_test_data, HEAD_labels, TAIL_labels = load_data(ARGS.dataset, data_setting=ARGS.data_setting, imbalanced_ratio=ARGS.imbalanced_ratio, head_sample_num=N_SAMPLES, min_valid_data_num=ARGS.min_valid_data_num)

if ARGS.data_augment:
    if ARGS.gmodel == 'bart':
        assert ARGS.cmodel == None

    augment_data_filename = './data/%s/aug_%s_%s_%s.csv' % (ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.gmodel)
    augment_data_filename = './data/%s/aug_%s_%s_%s.csv' % (ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.gmodel)
    
    print('data_augment:', ARGS.data_augment)
    print('cmodel:', ARGS.cmodel)
    print('gmodel:', ARGS.gmodel)
    
    labeled_aug_data = augment_data(augment_data_filename, tokenizer)
    labeled_train_data = pd.concat([labeled_aug_data, labeled_train_data])


## 
#N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES
#if ARGS.imb_type == 'longtail':
#    N_SAMPLES_PER_CLASS_BASE = make_longtailed_imb(N_SAMPLES, N_CLASSES, ARGS.ratio)
#elif ARGS.imb_type == 'step':
#    for i in range(ARGS.imb_start, N_CLASSES):
#        N_SAMPLES_PER_CLASS_BASE[i] = int(N_SAMPLES * (1 / ARGS.ratio))
#elif ARGS.imb_type == 'all':
#    for i in range(ARGS.imb_start, N_CLASSES):
#        N_SAMPLES_PER_CLASS_BASE[i] = -1

#N_SAMPLES_PER_CLASS_BASE = tuple(N_SAMPLES_PER_CLASS_BASE)
#print(N_SAMPLES_PER_CLASS_BASE)
###

df_train_data_filtered = labeled_train_data[['sentence', 'label']]
df_train_sent_group_by_label = df_train_data_filtered.groupby(['label']).size().reset_index(name='counts')
classes, class_counts = np.unique(df_train_sent_group_by_label['label'].tolist(), return_counts=True)

##
N_SAMPLES_PER_CLASS = tuple(df_train_sent_group_by_label.counts.tolist())
N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)
print(N_SAMPLES_PER_CLASS)
#print(N_SAMPLES_PER_CLASS_T)
##

def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0,length))  
#     tmp_l = []
#     for i in range(0, length):
#         _, _, _, label = dataset.__getitem__(i)
#         label = label[0]
#         tmp_l.append(label)
#     print(set(tmp_l))
#     assert 1 == 2
    
    for i in range(0, length):
        index = indices[i]
        _, _, _, label = dataset.__getitem__(index)
        label = label[0]
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_data_dict(x_dict, y_dict, train=False, shuffle=False, skip=False):
    for t in ['train', 'valid', 'test']:
        x_dict['HEAD_'+t], x_dict['TAIL_'+t], y_dict['HEAD_'+t], y_dict['TAIL_'+t] = split_HEAD_TAIL(x_dict[t], y_dict[t])
   
    data_dict = {}
    loader_dict = {}

    # JW (temp)
    iter_list = ['', 'HEAD', 'TAIL']
    if skip: iter_list = ['']

    for t1 in iter_list:
        for t2 in ['train', 'valid', 'test']:
            key = t2
            if t1 != '':
                key = t1+'_'+key
            data_dict[key] = Dataset(x_dict[key], y_dict[key])
            if train or t2 != 'train':
                if t2 == 'train':
                    from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
                    train_in_idx = get_oversampled_data(data_dict[key], num_of_class_samples, ARGS.random_seed)
                    loader_dict[key] = DataLoader(dataset=data_dict[key], 
                                                  sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), 
                                                  batch_size=ARGS.batch_size, collate_fn=collate_fn, shuffle=False)
                else:
                    loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=ARGS.batch_size, collate_fn=collate_fn, shuffle=False)
    return x_dict, y_dict, data_dict, loader_dict


x_dict, y_dict, y_class, class_dict = convert_data(labeled_train_data, labeled_valid_data, labeled_test_data, tokenizer)

num_of_classes = len(y_class)
num_of_class_samples = [0] * (max(y_class) + 1)
max_samples = df_train_sent_group_by_label['counts']
max_count = max(df_train_sent_group_by_label['counts'].tolist())
for index, row in df_train_sent_group_by_label.iterrows():        
    num_of_class_samples[row.label] = row.counts    
num_of_class_samples_T = torch.tensor(num_of_class_samples, dtype=torch.float)

x_dict, y_dict, data_dict, loader_dict = get_data_dict(x_dict, y_dict, train=True, shuffle=True)

train_loader, val_loader, test_loader = loader_dict['train'], loader_dict['valid'], loader_dict['test']




if 'translation.py' in sys.argv[0] or 'data_generation_for_gmodel.py' in sys.argv[0]:
    from fairseq.models.bart import BARTModel
    from transformers import BartTokenizer
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
elif 'token_importance.py' in sys.argv[0] == False:
    print('Tokenizer is deleted!')
    del tokenizer

torch.cuda.empty_cache()
