import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn

from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

import random
import sys
from collections import defaultdict
import pandas as pd

from tqdm import tqdm

#Global variables
def set_global_variables(_gpu=True, _device=0, _MAX_LEN=512, _batch_size=128, _HEAD_labels=[0, 9, 10, 17, 19, 25, 32, 40, 46, 49, 51, 52, 56]):
    global gpu
    global device
    global MAX_LEN
    global batch_size
    global HEAD_labels
    gpu = _gpu
    device = _device
    MAX_LEN = _MAX_LEN
    batch_size = _batch_size
    HEAD_labels = _HEAD_labels

    def gpu_settings(device):
        if gpu:
            torch.cuda.set_device(device)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            CUDA_LAUNCH_BLOCKING=device
    gpu_settings(_device)


#Functions
#load_data, get_bert, get_labeled_data, get_unlabeled_data, Dataset, collate_fn, CosSimTanhModel, split_HEAD_TAIL

def load_data():
    print('load_data: start')
    labeled_data = []
    max_sent_length = 0
    with open('factor_labeled_data_full', 'r', encoding='utf8') as fp:
        lines = fp.readlines()
        for line in lines:
            json_data = json.loads(line.split('\n')[0])
            if 'factor_label' not in list(json_data['factor_label'][0].keys()) or json_data['factor_label'][0]['factor_label'] == []: continue
            sent = json_data['sentence']
            max_sent_length = max(max_sent_length, len(sent))
            label = json_data['factor_label'][0]['factor_label'][0].split('/')[0]
            labeled_data.append((sent, label))
    
    print('# of sents:', len(labeled_data))
    print('max length of sents:', max_sent_length)
    print('load_data: done')
    return labeled_data

def save_labeled_data(labeled_data, filename):
    fwrite = open(filename, 'w')
    for l in labeled_data:
        fwrite.write(l['labeling_rule'], l['sentence'] + '\n')
    fwrite.close()

def get_labeled_data(labeled_data):
    print('get_labeled_data: start')
    df_sent = pd.DataFrame(columns=['sentence', 'label', 'data_source'])
    for sent, label in tqdm(labeled_data):
        df_sent = df_sent.append({'sentence': sent, 'label': label, 'data_source':'NC'}, ignore_index=True)
    
    print('get_labeled_data: done')
    return df_sent

def generate_M2M_data(df_label, df_sent):
    # parameters (need to be replaced by a config file)
    filename_x = './data/NC_M2M/m2m_x_data'
    filename_test_x = './data/NC_M2M/m2m_test_x_data'

    major_train_sample_ratio = 0.5 # TBD
    major_test_sample_ratio = 0.5 # TBD
    minor_test_ratio = 1.0 # TBD
    is_shuffle = False

    fwrite_x = open(filename_x, 'w')
    fwrite_test_x = open(filename_test_x, 'w')
    
    # split training & test sentences for each label
    label_sent_train_dict = defaultdict(list)
    label_sent_test_dict = defaultdict(list)
    
    for i in range(len(df_label)):
        label_name = df_label.iloc[i]['label']
        label_sent = df_sent.loc[df_sent['label'] == df_label.iloc[i]['label']]['sentence'].to_list()

        if i in _HEAD_labels:        
            label_sent_train_dict[label_name] = label_sent[:len(label_sent)//2]
            label_sent_test_dict[label_name] = label_sent[len(label_sent)//2:]

        if i not in _HEAD_labels:        
            label_sent_train_dict[label_name] = label_sent
            label_sent_test_dict[label_name] = []
   
    
    # generate training & test data for each label
    for i in range(len(df_label)):
        if i not in _HEAD_labels: continue
        label_name = label_name = df_label.iloc[i]['label']
        # training data
        for j in range(i, len(df_label)): # including the same label
            for k in range(min(500, len(label_sent_train_dict[label_name]))):                
                label_name2 = df_label.iloc[j]['label']
                x = random.choice(label_sent_train_dict[label_name])
                y = random.choice(label_sent_train_dict[label_name2])
                
                fwrite_x.write(str(i) + '\t' + str(j) + '\t' + x + '\t' + y + '\n')

            # test data
            if j in _HEAD_labels: continue
            for x in label_sent_test_dict[label_name][:10]:
                fwrite_test_x.write(str(i) + '\t' + str(j) + '\t' + x + '\n')
     
    fwrite_x.close()
    fwrite_test_x.close()

def print_df_csv(filename, df):
    df.to_csv(filename, index=False)
    return

def df_filter(df_sent):
    df_label = pd.DataFrame(columns=['label'])

    df_label['label'] = sorted(df_sent.label.unique())
    df_label['is_head'] = False

    #_HEAD_labels = [0, 1] # TBR
    df_label.iloc[_HEAD_labels, [1]] = True

    df_label['size'] = df_sent.groupby(['label']).size().tolist()
    df_label_filtered = df_label[df_label['size'] >= 10]
    df_label_filtered['id'] = range(0, len(df_label_filtered))

    df_sent_filtered = df_sent[df_sent['label'].isin(df_label_filtered['label'])]
    df_sent_filtered['id'] = range(0, len(df_sent_filtered))

    df_label_filtered = df_label_filtered[['id', 'label', 'is_head', 'size']]
    df_sent_filtered = df_sent_filtered[['id', 'sentence', 'label', 'data_source']]

    print(df_sent.groupby(['label']).size().to_string())
    #df_sent.groupby(['label']).size()#.plot(kind='bar')
    print ('DF', len(df_sent), df_sent.head(9))
    print ('DF_Filtered', len(df_sent_filtered), df_sent_filtered.head(9))
    print ('DF_Label', df_label)
    print ('DF_Label_Filtered', df_label_filtered)
    return df_label_filtered, df_sent_filtered

if __name__ == '__main__':    
    random.seed(0)
    global _HEAD_labels
    _HEAD_labels=[0, 9, 10, 17, 19, 25, 32, 40, 46, 49, 51, 52, 56]


    filename = './data/NC_M2M/m2m_labled_data'
    labeled_data = load_data() 
    df_sent = get_labeled_data(labeled_data)
    df_label, df_sent = df_filter(df_sent)
    print_df_csv('./data/NC_M2M/label.csv', df_label)
    print_df_csv('./data/NC_M2M/sent.csv', df_sent)
    generate_M2M_data(df_label, df_sent)
    print ('# of labels', len(df_label['label']))
