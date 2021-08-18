import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import random
import sys  

#Global variables
# Need to be update!
def set_global_variables(_gpu=True, _device=0, _MAX_LEN=128, _batch_size=128, _HEAD_labels=[0, 9, 10, 17, 19, 25, 32, 40, 46, 49, 51, 52, 56]):
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
    
# Model utils
def get_bert():
    model = BertModel.from_pretrained('bert-base-uncased')    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    
    return model, tokenizer

# Functions
# load_data, get_bert, get_labeled_data, get_unlabeled_data, Dataset, collate_fn, CosSimTanhModel, split_HEAD_TAIL
# Data utils
# make a class with data manipulation functions
def data_factory(data_source, tokenizer):
    labeled_train_data, labeled_valid_data, labeled_test_data = load_data(data_source)
    x_dict, y_dict, y_class, class_dict = convert_data(labeled_train_data, labeled_valid_data, labeled_test_data, tokenizer)
    return x_dict, y_dict, y_class, class_dict


def load_data(data_source, data_setting=None, imbalanced_ratio=0,head_sample_num=0,min_valid_data_num=0): # load data always returns train, valid, test dataset
    if data_source=='ATIS':        
        return load_atis_data()
    elif data_source=='TREC':
        return load_trec_data(data_setting=data_setting, imbalanced_ratio=imbalanced_ratio, head_sample_num=head_sample_num, min_valid_data_num=min_valid_data_num)
    elif data_source=='SNIPS':
        return load_snips_data(data_setting=data_setting, imbalanced_ratio=imbalanced_ratio, head_sample_num=head_sample_num, min_valid_data_num=min_valid_data_num)
    elif data_source=='SENT':
        return load_sent_data(data_setting=data_setting, imbalanced_ratio=imbalanced_ratio)
    else:
        raise (NameError('Allowing the following sources: ["ATIS", "TREC"]'))


def augment_data(filename, tokenizer):
    sents = []
    labels = []

    for line in open(filename, 'r', encoding='utf-8'):
       splits = line.strip().split('\t')
       if len(splits) != 2: continue # needs to be handdled in tranlsation (or generation)
       sents.append(splits[1].strip())
       labels.append(int(splits[0]))

    df = pd.DataFrame(list(zip(sents, labels)), columns=['sentence', 'label'])
    return df   


# Needs to apply template method pattern
# This method needs to be overrided for each dataset
def load_atis_data():
    global HEAD_labels
    global TAIL_labels

    DATA_DIR = './data/ATIS/'
    FILTER_SAMPLE_CONDITION = 5
    HEAD_SAMPLE_CONDITION = 100

    # Load Data
    labeled_train_data = pickle.load(open(DATA_DIR + 'atis_labeled_train_data.pkl', 'rb'))    
    labeled_test_data = pickle.load(open(DATA_DIR + 'atis_labeled_test_data.pkl', 'rb'))

    # Split Data
    labeled_train_data, labeled_valid_data = split_data(labeled_train_data, [0.9, 0.1], class_wise=True)                

    # Filter Data (Five shot condition)
    selected_labels = get_five_sample_labels(labeled_train_data, filter_size=FILTER_SAMPLE_CONDITION)
    labeled_train_data_filtered = select_with_labels(labeled_train_data, selected_labels)
    labeled_valid_data_filtered = select_with_labels(labeled_valid_data, selected_labels)
    labeled_test_data_filtered = select_with_labels(labeled_test_data, selected_labels)

    # Split HEAD Tail 
    HEAD_labels, TAIL_labels = get_head_and_tail_labels(labeled_train_data_filtered, head_size=HEAD_SAMPLE_CONDITION)

    return labeled_train_data_filtered, labeled_valid_data_filtered, labeled_test_data_filtered, HEAD_labels, TAIL_labels

def load_trec_data(data_setting=None, imbalanced_ratio=0, head_sample_num=0, min_valid_data_num=0):
    global HEAD_labels
    global TAIL_labels

    DATA_DIR = './data/TREC/'
    FILTER_SAMPLE_CONDITION = 5
    HEAD_SAMPLE_CONDITION = 100
    num_class = 6

    # Load Data
    labeled_train_data = pickle.load(open(DATA_DIR + 'trec_labeled_train_data.pkl', 'rb'))    
    labeled_test_data = pickle.load(open(DATA_DIR + 'trec_labeled_test_data.pkl', 'rb'))

    # Imbalanced Data Manipulation
    if data_setting == 'all':
        print('No Imbalanced Data Mainpulation')

        # Split Data
        labeled_train_data, labeled_valid_data = split_data(labeled_train_data, [0.9, 0.1], class_wise=True, min_valid_data_num=min_valid_data_num)                

    elif data_setting == 'longtail' or data_setting == 'step':
        assert imbalanced_ratio > 0
        labeled_train_data, labeled_valid_data, HEAD_SAMPLE_CONDITION = get_imbalanced_data(labeled_train_data, HEAD_SAMPLES=head_sample_num, dataset='TREC', data_setting=data_setting, imbalanced_ratio=imbalanced_ratio, num_class=num_class, min_valid_data_num=min_valid_data_num)
    else:
        raise NotImplmentError(data_setting)

    print(labeled_train_data)
    # Filter Data (Five shot condition)
    selected_labels = get_five_sample_labels(labeled_train_data, filter_size=FILTER_SAMPLE_CONDITION)
    labeled_train_data_filtered = select_with_labels(labeled_train_data, selected_labels)
    labeled_valid_data_filtered = select_with_labels(labeled_valid_data, selected_labels)
    labeled_test_data_filtered = select_with_labels(labeled_test_data, selected_labels)

    # Split HEAD Tail 
    HEAD_labels, TAIL_labels = get_head_and_tail_labels(labeled_train_data, head_size=HEAD_SAMPLE_CONDITION)
    print('HEAD_LABELS: %s, Tail_labels %s, HEAD_SAMPLE_CONDTION: %s' % (HEAD_labels, TAIL_labels, HEAD_SAMPLE_CONDITION))

    return labeled_train_data_filtered, labeled_valid_data_filtered, labeled_test_data_filtered, HEAD_labels, TAIL_labels


def load_snips_data(data_setting=None, imbalanced_ratio=0, head_sample_num=0, min_valid_data_num=0):
    global HEAD_labels
    global TAIL_labels

    DATA_DIR = './data/SNIPS/'
    FILTER_SAMPLE_CONDITION = 0
    HEAD_SAMPLE_CONDITION = 1800
    num_class = 7

    # Load Data
    labels, sentences = parse_tsv_file(DATA_DIR + 'train.tsv')
    labeled_train_data = pd.DataFrame(list(zip(labels, sentences)), columns=['label', 'sentence'])
    
    labels, sentences = parse_tsv_file(DATA_DIR + 'test.tsv')
    labeled_test_data = pd.DataFrame(list(zip(labels, sentences)), columns=['label', 'sentence'])

    # Imbalanced Data Manipulation    
    if data_setting == None or data_setting == 'all':
        print('No Imbalanced Data Mainpulation')

        # Split Data
        labeled_train_data, labeled_valid_data = split_data(labeled_train_data, [0.9, 0.1], class_wise=True, min_valid_data_num=min_valid_data_num) 

    elif data_setting == 'longtail' or data_setting == 'step':
        assert imbalanced_ratio > 0        
        labeled_train_data, labeled_valid_data, HEAD_SAMPLE_CONDITION = get_imbalanced_data(labeled_train_data, HEAD_SAMPLES=head_sample_num, dataset='SNIPS', data_setting=data_setting, imbalanced_ratio=imbalanced_ratio, num_class=num_class, min_valid_data_num=min_valid_data_num)

    else:
        raise NotImplmentError(data_setting)


    # Filter Data (Five shot condition)
    selected_labels = get_five_sample_labels(labeled_train_data, filter_size=FILTER_SAMPLE_CONDITION)
    labeled_train_data_filtered = select_with_labels(labeled_train_data, selected_labels)
    labeled_valid_data_filtered = select_with_labels(labeled_valid_data, selected_labels)
    labeled_test_data_filtered = select_with_labels(labeled_test_data, selected_labels)

    # Split HEAD Tail 
    HEAD_labels, TAIL_labels = get_head_and_tail_labels(labeled_train_data, head_size=HEAD_SAMPLE_CONDITION)
    print('HEAD_LABELS: %s, Tail_labels %s, HEAD_SAMPLE_CONDTION: %s' % (HEAD_labels, TAIL_labels, HEAD_SAMPLE_CONDITION))

    return labeled_train_data_filtered, labeled_valid_data_filtered, labeled_test_data_filtered, HEAD_labels, TAIL_labels


def load_sent_data(data_setting=None, imbalanced_ratio=0):
    global HEAD_labels
    global TAIL_labels

    def parse_sent_tsv_file(filename):# format: [label] \t [sentence]
        labels, sentences = [], []
        for line in open(filename, 'r', encoding='utf-8').readlines():
            splits = line.strip().split('\t')
            label, sentence = splits[0], splits[1]
            if label == 'Negative':
                labels.append(0)
            elif label == 'Positive':
                labels.append(1)
            else:
                continue
            sentences.append(sentence)    

        return labels, sentences    

    def get_df(path):
        labels, sentences = parse_sent_tsv_file(path)
        return pd.DataFrame(list(zip(labels, sentences)), columns=['label', 'sentence'])

    #path = './data/counterfactually-augmented-data/sentiment/combined/'
    path = './data/SENT/'
    labeled_train_data = get_df(path+'train_paired.tsv')
    labeled_valid_data = get_df(path+'dev_paired.tsv')
    labeled_test_data = get_df(path+'test_paired.tsv')
    
    if data_setting != 'all':
        pos_train = labeled_train_data[labeled_train_data['label'] == 1]
        neg_train = labeled_train_data[labeled_train_data['label'] == 0]

        minor_idx = list(range(len(neg_train)))
        random.shuffle(minor_idx)
        minor_idx = minor_idx[:int(len(pos_train)/100)]

        labeled_train_data = pos_train.append(neg_train.iloc[minor_idx])

    # Split HEAD Tail 
    HEAD_SAMPLE_CONDITION = 1707
    HEAD_labels, TAIL_labels = get_head_and_tail_labels(labeled_train_data, head_size=HEAD_SAMPLE_CONDITION)
    print('HEAD_LABELS: %s, Tail_labels %s, HEAD_SAMPLE_CONDTION: %s' % (HEAD_labels, TAIL_labels, HEAD_SAMPLE_CONDITION))
        
    return labeled_train_data, labeled_valid_data, labeled_test_data, HEAD_labels, TAIL_labels


def parse_tsv_file(filename): # format: [label] \t [sentence]
    labels, sentences = [], []

    for line in open(filename, 'r', encoding='utf-8').readlines():
        splits = line.strip().split('\t')
        label, sentence = splits[0], splits[1]
        labels.append(int(label))
        sentences.append(sentence)    

    return labels, sentences


def get_imbalanced_data(labeled_train_data, HEAD_SAMPLES=None, dataset=None, data_setting=None, imbalanced_ratio=None, num_class=None, min_valid_data_num=0):
    print('Imbalanced Data Manipulation: %s Imbalance (Imbalanced ratio: %s)' % (data_setting, str(imbalanced_ratio)))        

    n_sample_per_class = get_sample_number_per_class(labeled_train_data, head_sample_number=HEAD_SAMPLES, dataset=dataset, data_setting=data_setting, imbalanced_ratio=imbalanced_ratio, num_class=num_class)
    head_sample_condition = sorted(n_sample_per_class, key=lambda x: -x)[num_class // 2 - 1]
    print('N_sample_per_class:', n_sample_per_class)

    labeled_train_data_sampled, labeled_valid_data_sampled = sample_imbalanced_data(labeled_train_data, n_sample_per_class, min_valid_data_num=min_valid_data_num)
    print('Sampled labeled train data num: %s, valid data num: %s' % (str(labeled_train_data_sampled.shape), str(labeled_valid_data_sampled.shape)))

    return labeled_train_data_sampled, labeled_valid_data_sampled, head_sample_condition

def sample_imbalanced_data(labeled_train_data, n_sample_per_class, min_valid_data_num=0, random_seed=7777): 
    labeled_train_data_sampled = pd.DataFrame(columns=['label', 'sentence'])
    labeled_valid_data_sampled = pd.DataFrame(columns=['label', 'sentence'])

    for i, label in enumerate(range(len(n_sample_per_class))):
        train_data_for_label = labeled_train_data.loc[labeled_train_data['label'] == label]
        target_valid_num = max(round(n_sample_per_class[i] * 0.1), min_valid_data_num)

        df_sampled = train_data_for_label.sample(n=round(n_sample_per_class[i] + target_valid_num), replace=False, random_state=random_seed)
        df_train_sampled = df_sampled[:n_sample_per_class[i]] 
        df_valid_sampled = df_sampled[n_sample_per_class[i]:]

        labeled_train_data_sampled = pd.concat([labeled_train_data_sampled, df_train_sampled])
        labeled_valid_data_sampled = pd.concat([labeled_valid_data_sampled, df_valid_sampled])

    return labeled_train_data_sampled, labeled_valid_data_sampled


def get_sample_number_per_class(labeled_train_data, head_sample_number=None, dataset=None, data_setting=None, imbalanced_ratio=None, num_class=None):    
    n_sample_per_class = [head_sample_number] * num_class

    # For TREC data, we change the sequence of classes in decending order (# of samples) because its label 0 is a minor class.
    class_indexes = [] 
    if dataset == 'TREC' or dataset == 'SNIPS': 
        sample_counts_per_class = labeled_train_data.groupby(['label']).size().tolist()
        class_indexes = np.argsort(sample_counts_per_class)[::-1]  # sorted by decending order
    #elif dataset == 'SNIPS':
    #    class_indexes = [i for i in range(num_class)]
    else:
        raise NotImplementedError()
    
    if data_setting == 'longtail':
        mu = np.power(1/imbalanced_ratio, 1/(num_class - 1))
        for i in range(num_class):
            n_sample_per_class[class_indexes[i]] = int(n_sample_per_class[class_indexes[i]] * np.power(mu, i))

    elif data_setting == 'step':
        imb_start = num_class // 2
        for i in range(imb_start, num_class):
            n_sample_per_class[class_indexes[i]] = n_sample_per_class[class_indexes[i]] // imbalanced_ratio

    return n_sample_per_class

# Reusable methods for all dataset
def get_five_sample_labels(labeled_train_data, filter_size=5):
    df_label = labeled_train_data.groupby(['label']).size().reset_index(name='size') 
    df_label = df_label.loc[df_label['size'] >= filter_size]
    return df_label.label.tolist()


def select_with_labels(labeled_data, label_list):
    return labeled_data[labeled_data['label'].isin(label_list)]


def get_head_and_tail_labels(labeled_train_data, head_size=50):
    df_label = labeled_train_data.groupby(['label']).size().reset_index(name='size')
    df_head_label = df_label.loc[df_label['size'] >= head_size]
    df_tail_label = df_label.loc[df_label['size'] < head_size]
#     df_tail_label = df_label.loc[(df_label['size'] > 0)&(df_label['size'] < head_size)]
    tail_labels = df_tail_label.label.tolist()
    print(tail_labels)
    return df_head_label.label.tolist(), tail_labels


# To be refactored
def split_data(source_df, ratio_list, class_wise=False, random_seed = 7777, min_valid_data_num=0):     
    assert sum(ratio_list) == 1.0
    data_tuple = []

    # accumulate ratio
    ratio_acc_list = np.add.accumulate(ratio_list)
    
    if class_wise:    
        class_sample_size_list = np.array(source_df.groupby(['label']).size())
        class_ratio_acc_matrix = np.array(class_sample_size_list[:, np.newaxis] * ratio_acc_list, dtype=int)        
        y_class = np.unique(source_df['label'])        
        
        for i, label in enumerate(y_class):
            source_df_for_label = source_df.loc[source_df['label'] ==label]

            # assert that groupby(['label']).size()) and np.unique(source_df['label'] return the same results (i.e., same order)            
            assert len(source_df_for_label) == class_ratio_acc_matrix[i][-1]
            
            # split data according to ratio 
            splits = np.split(source_df_for_label.sample(frac=1, random_state=random_seed), class_ratio_acc_matrix[i][:-1])
            
            # To be refactored
            #splits[0].index) < min_valid_data_num: # DEV
            while len(splits[0].index) > 0 and len(splits[1].index) < min_valid_data_num:
                sample = splits[0].sample()
                splits[1] = splits[1].append(sample, ignore_index=True)
                splits[0].drop(sample.index, inplace=True)

            for j, split in enumerate(splits):                
                if i == 0:
                    data_tuple.append(split)
                else:
                    data_tuple[j] = pd.concat([data_tuple[j], split])

    elif class_wise == False:  
        ratio_acc_list = np.array(ratio_acc_list * len(source_df), dtype=int)
        data_tuple = np.split(source_df.sample(frac=1, random_state=random_seed), ratio_acc_list)
        
    return data_tuple


def extract_feature(labeled_data, tokenizer):    
    labeled_data['input_ids'] = labeled_data['sentence'].apply(lambda sent: np.array(tokenizer.encode(sent, add_special_tokens=True)))  
    labeled_data['input_mask'] = labeled_data['input_ids'].apply(lambda input_ids: np.ones(input_ids.shape, np.long))
    labeled_data['token_type_ids'] = labeled_data['input_ids'].apply(lambda input_ids: np.zeros(input_ids.shape, np.long))

    #labeled_data['input_ids'] = labeled_data['input_ids'].apply(lambda input_ids : np.concatenate([input_ids, np.zeros(MAX_LEN - len(input_ids))]).astype(int))  
    #labeled_data['input_mask'] = labeled_data['input_mask'].apply(lambda input_mask : np.concatenate([input_mask, np.zeros(MAX_LEN - len(input_mask))]).astype(int))  
    #labeled_data['token_type_ids'] = labeled_data['token_type_ids'].apply(lambda token_type_ids : np.concatenate([token_type_ids, np.zeros(MAX_LEN - len(token_type_ids))]).astype(int))  
    
    return labeled_data


def convert_data(labeled_train_data, labeled_valid_data, labeled_test_data, tokenizer):
    x_dict, y_dict, class_dict = dict(), dict(), dict()
    num_of_class = np.unique(labeled_train_data['label'])
    
    y_class = [i for i in range(max(labeled_train_data['label'])+ 1)]        
    
    labeled_train_data = extract_feature(labeled_train_data, tokenizer)
    labeled_test_data = extract_feature(labeled_test_data, tokenizer)
    labeled_valid_data = extract_feature(labeled_valid_data, tokenizer)
    
    x_dict['train'] = labeled_train_data[['input_ids', 'input_mask', 'token_type_ids']].to_records(index=False).tolist()
    y_dict['train'] = labeled_train_data[['label']].values
    x_dict['valid'] = labeled_valid_data[['input_ids', 'input_mask', 'token_type_ids']].to_records(index=False).tolist()
    y_dict['valid'] = labeled_valid_data[['label']].values
    x_dict['test'] = labeled_test_data[['input_ids', 'input_mask', 'token_type_ids']].to_records(index=False).tolist()
    y_dict['test'] = labeled_test_data[['label']].values
    #print(labeled_test_data[['label']].values)

    return x_dict, y_dict, y_class, class_dict


class Dataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        input_ids, input_mask, token_type_ids = self.X[index]
        #print('Y', self.Y[index])
        return input_ids, input_mask, token_type_ids, self.Y[index]
    def __len__(self):
        return self.len

def collate_fn(batch):
    _input_ids, _input_mask, _token_type_ids, _label = zip(*batch)
    
    max_len = 0
    for sent in _input_ids:
        if max_len < len(sent):
            max_len = len(sent)
    max_len = min(max_len, MAX_LEN)
    
    bs = min(batch_size, len(_input_ids))
    
    input_ids = np.zeros([bs, max_len], np.long)
    input_mask = np.zeros([bs, max_len], np.long)
    token_type_ids = np.zeros([bs, max_len], np.long)
    label = np.zeros([bs], np.long)
    
    for i in range(bs):
        for j in range(len(_input_ids[i])):
            if j >= max_len:
#                 max_len_count += 1
                break
            input_ids[i, j] = _input_ids[i][j]
            input_mask[i, j] = _input_mask[i][j]
            token_type_ids[i, j] = _token_type_ids[i][j]
        #label[i, _label[i]] = 1
        label[i] = _label[i]
    
    batch = {}
    batch['input_ids'] = input_ids
    batch['input_mask'] = input_mask
    batch['token_type_ids'] = token_type_ids
    batch['label'] = label
    
    for key in batch.keys():
        batch[key] = torch.tensor(np.asarray(batch[key]))
        if gpu:
            batch[key] = batch[key].cuda()    
    return batch

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class CosSimModel(nn.Module):
    def __init__(self, num_of_class):
        super(CosSimModel, self).__init__()
        np.random.seed(7777)
        torch.manual_seed(7777)
        torch.cuda.manual_seed_all(7777)
        random.seed(7777)
        self.bert, _, _ = get_bert()
        for param in self.bert.parameters():
            param.requires_grad = False
        self.num_of_class = num_of_class
        self.frame()
        self.score = None
        self.name = 'baseline.ckpt'
        
    def frame(self):
        self.linear = nn.Linear(768, 768)
        self.w_k = nn.Linear(768, self.num_of_class)
        self.tau = torch.nn.Parameter(torch.cuda.FloatTensor([10]))
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def get_norm2(self, tensor, dim=1):
        norm2 = torch.norm(tensor, p=2, dim=dim).detach()
        if gpu:
            norm2 = norm2.cuda()
        if dim == 0:
            return norm2.view(1, -1)
        elif dim == 1:
            return norm2.view(-1, 1)
        else:
            raise NotImplementedError

    def get_repr(self, input_ids, token_type_ids, input_mask):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, input_mask)
        z = self.linear(pooled_output)
        return z
    
    def get_score(self, vec1, vec2):
        #return torch.mm(self.tanh(vec1), self.tanh(vec2)) * self.tau
        return torch.mm(vec1 / self.get_norm2(vec1), (vec2 / self.get_norm2(vec2)).t()) * self.tau        
            
    def forward(self, input_ids, token_type_ids, input_mask):
        z = self.get_repr(input_ids, token_type_ids, input_mask)        
        return self.get_score(z, self.w_k.weight)
        
    def criterion(self, output, label):
        #output= output.double()
        loss_fn = nn.NLLLoss(reduction='none')
        loss = loss_fn(self.logsoftmax(output), label)

        return loss.sum()
    
    #def _train(self, train_data, valid_data, test_data, valid_data2, test_data2, lr=1e-5, wd=1e-6, save=True, num_of_epoch=1, name=None):
    def _train(self, train_data, valid_data, test_data, lr=1e-5, wd=1e-6, save=True, num_of_epoch=1, name=None):
        self.global_step = 0
        self.total_steps = 454 * num_of_epoch
        self.eval_period = int(len(train_data) * 0.1)+1
        if name is None:
            name = self.name
        self.load(name)
        self.train()
        #self.bert.eval()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        tmp_best_score = None
        #print(len(train_data))
        for _ in range(num_of_epoch):
            for i, batch in tqdm(enumerate(train_data)):
                output = self.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids'])
                loss = self.criterion(output, batch['label'])#.sum()
                self.zero_grad()
                loss.backward()
                optimizer.step()
                if i % self.eval_period == 0:
                    print('EVAL')
                    tmp_score = self.evaluation(valid_data)
                    if tmp_best_score is None or tmp_best_score < tmp_score:
                        tmp_best_score = tmp_score
                    if (self.score is None or self.score < tmp_score) and save:
                        torch.cuda.empty_cache()
                        test = self.evaluation(test_data)
#                         all_valid = self.evaluation(valid_data2)
#                         all_test = self.evaluation(test_data2)
#                         print('RARE')
                        print("VALID:", round(tmp_score, 4), '\tTEST:', round(test, 4))
#                         print('ALL')
#                         print("VALID:", round(all_valid, 4), '\tTEST:', round(all_test, 4))
                        self.score = tmp_score
                        self.save(name)
                    else:
                        print("VALID:", round(tmp_score, 4), "[NOSAVE]")
                torch.cuda.empty_cache()
        self.eval()
        #print("%.4f" % tmp_best_score)
        return tmp_best_score
    
    
    def evaluation(self, data):
        self.eval()
        pred = None
        #for batch in tqdm(data):
        for batch in data:
            p = torch.argmax(self.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids']), dim=1).cpu().numpy()
            if pred is None:
                pred = p
            else:
                pred = np.concatenate([pred, p], axis=0)
        l = None
        for batch in data:
            _l = batch['label'].cpu().numpy()
            if l is None:
                l = _l
            else:
                l = np.concatenate([l, _l], axis=0)
        #rare_class = 
        #common_class = 
        self.train()
        #self.bert.eval()
        return (pred == l).astype(np.float32).sum() / l.shape[0]    
        
    def save(self, name=None):
        if name is None:
            name = self.name
        torch.save({'state_dict': self.state_dict()}, 'ckpt/'+name)
        
    def load(self, name=None):
        if name is None:
            name = self.name
        if not os.path.exists('ckpt/'+name):
            print('No ckpt')
            return
        ckpt = torch.load('ckpt/'+name, map_location='cuda:'+str(torch.cuda.current_device()))
        self.load_state_dict(ckpt['state_dict'])
        print(name, 'loaded')

        
class CosSimTanhModel(CosSimModel):
    def __init__(self, num_of_class):
        super().__init__(num_of_class)
        
    def get_score(self, vec1, vec2):
        return torch.mm(self.tanh(vec1), self.tanh(vec2).t()) * self.tau
        #return torch.mm(vec1 / self.get_norm2(vec1), (vec2 / self.get_norm2(vec2)).t()) * self.tau            
        
        
def split_HEAD_TAIL(x, y):
    HEAD_x, TAIL_x, HEAD_y, TAIL_y = [], [], [], []
    for _x, _y in zip(x, y):
        if _y in HEAD_labels:
            HEAD_x.append(_x)
            HEAD_y.append(_y)
        else:
            TAIL_x.append(_x)
            TAIL_y.append(_y)
    return HEAD_x, TAIL_x, HEAD_y, TAIL_y


def get_data_dict(x_dict, y_dict, train=False, shuffle=False, skip=False, nclasses=None, balanced_sampling=False):
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
                    if balanced_sampling:
                        def make_weights_for_balanced_classes(images, nclasses):                        
                            count = [0] * nclasses                                                      
                            for item in images:         
                                count[item[0]] += 1  
                            weight_per_class = [0.] * nclasses                                      
                            N = float(sum(count))                                                   
                            for i in range(nclasses):
                                if count[i] == 0:
                                    continue
                                weight_per_class[i] = N/float(count[i])                                 
                            weight = [0] * len(images)                                              
                            for idx, val in enumerate(images):                                          
                                weight[idx] = weight_per_class[val[0]]                                  
                            return weight
                        weights = make_weights_for_balanced_classes(y_dict[key], nclasses)          
                        weights = torch.DoubleTensor(weights)
                        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
                    else:
                        sampler = None
                        
                    loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=batch_size, collate_fn=collate_fn, sampler = sampler)
                else:
                    loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return x_dict, y_dict, data_dict, loader_dict

def get_data_dict_over(x_dict, y_dict, train=False, shuffle=False, skip=False):
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
                    loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
                else:
                    loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return x_dict, y_dict, data_dict, loader_dict


from collections import defaultdict

def generate_M2M_data(filename_train, y_class, df_sent):
    # parameters (need to be replaced by a config file)    
    fwrite_train = open(filename_train, 'w')

    # split training & test sentences for each label
    label_sent_train_dict = defaultdict(list)
    for i in range(len(y_class)):
        label_sent = df_sent.loc[df_sent['label'] == i]['sentence'].to_list()        
        label_sent_train_dict[i] = label_sent

    # generate training & test data for each label
    for i in range(len(y_class)):
        if len(label_sent_train_dict[i]) == 0: continue
            
        # training data
        for x in label_sent_train_dict[i]:
            fwrite_train.write(str(i) + '\t' + x + '\n')

    fwrite_train.close()

def get_k_shot_dataframe(loader_dict, data_type, TAIL_labels, tokenizer):
    # HEAD
    labels, sentences = [], []
    for batches in loader_dict['HEAD_{0}'.format(data_type)]:
        for i in range(len(batches['input_ids'])):
            label = int(batches['label'][i])
            labels.append(label)            
            sentences.append(tokenizer.decode(batches['input_ids'][i], skip_special_tokens=True))
    
    # TAIL 
    for batches in loader_dict['sampled_{0}'.format(data_type)]:
        for i in range(len(batches['input_ids'])):
            label = int(batches['label'][i][0])
            if label not in TAIL_labels: continue                
            labels.append(label)
            sentences.append(tokenizer.decode(batches['input_ids'][i], skip_special_tokens=True))
    
    assert len(labels) == len(sentences)
    return pd.DataFrame(list(zip(labels, sentences)), columns=['label', 'sentence'])

def convert2binary(data, i, seed, train=True, aug=False):
#     with open('labeled_train_data.pkl', 'wb') as fp:
#         pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
    num_of_label = len(list(set(data['label'])))
    num_of_data = []
    for idx in range(7):
        num_of_data.append(len(data[data['label'] == idx]))
    print('\nlabel distribution', num_of_data)
    pos = data[data['label']==i]
    neg_data = data[data['label']!=i]
    if not train:
        neg = neg_data
        print('total data size:', len(data))
        print('pos data size:', len(pos))
        print('neg data size:', len(neg))
        pos['label'] = 1
        neg['label'] = 0
        return pd.concat([pos, neg])
    else:        
        assert aug or num_of_data[i] != max(num_of_data)
        random.seed(seed)
        neg_idx = random.sample(list(range(0,len(neg_data))), len(pos))
        neg = neg_data.iloc[neg_idx]
        neg_rest = neg_data.drop(list(neg.index))
        print('total data size:', len(data))
        print('pos data size:', len(pos))
        print('neg data size:', len(neg))
        print('neg_rest data size:', len(neg_rest))
        pos['label'] = 1
        neg['label'] = 0
        neg_rest['label'] = 0
        return pd.concat([pos, neg]), neg_rest

    
def save_loader(loader, fn):
    x = []
    y = None
    for batch in loader:
        _x = batch['input_ids'].cpu().numpy()
        _y = batch['label'].cpu().numpy()
        x.append(_x)
        if y is None:
            y = _y
        else:
            y = np.concatenate([y, _y], axis=0)
    with open(fn, 'wb') as fp:
        pickle.dump((_x, _y), fp, pickle.HIGHEST_PROTOCOL)
    print(fn+' [SAVED]')

        