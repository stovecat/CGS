import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn

from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader


import random

import sys  


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
    labeled_data = []
    with open('factor_labeled_data_full', 'r', encoding='utf8') as fp:
        lines = fp.readlines()
        for line in lines:
            labeled_data.append(json.loads(line.split('\n')[0]))
    with open('processed_news_sents.pkl', 'rb') as fp:
        processed_data = pickle.load(fp)
    return labeled_data, processed_data

def get_bert():
    model, vocab  = get_pytorch_kobert_model()
    tok_path = get_tokenizer()
    sp  = SentencepieceTokenizer(tok_path)
    return model, vocab, sp



def get_class_dict(l_list):
    dict = {}
    for i, l in enumerate(l_list):
        class_val = str(l)
        if class_val not in dict.keys():
            dict[class_val] = []
        dict[class_val].append(i)
    return dict

# 2020/6/9 - Jinwoo Par
# Added an optional parameter contain_sent for adding sents in x
def get_labeled_data(labeled_data, vocab, sp, contain_sent=False):
    _x = []
    _y = []
    y_set = set()
    num_of_class = 0
    for data in tqdm(labeled_data):
        sent = data['sentence']
        input_ids = np.array([vocab['[CLS]']]+ [vocab[t] for t in sp(sent)]+[vocab['[SEP]']])
        input_mask = np.ones(input_ids.shape, np.long)
        token_type_ids = np.zeros(input_ids.shape, np.long)
        if 'factor_label' not in list(data['factor_label'][0].keys()) or data['factor_label'][0]['factor_label'] == []:
            continue
        l = data['factor_label'][0]['factor_label'][0].split('/')[0]

        if contain_sent:
            _x.append((input_ids, token_type_ids, input_mask, sent))
        else: # original code without sents
            _x.append((input_ids, token_type_ids, input_mask))
        _y.append(l)

        if l not in y_set:
            y_set.add(l)
            num_of_class += 1
    y_class = sorted(list(tuple(y_set)))

    y = []
    for i, l in enumerate(_y):
        y.append(np.array(y_class.index(l)))
        
    test_size = int(len(y) * 0.05)
        
    def get_each_items(dict):
        items = []
        for i in range(len(y_class)):
            i = str(i)
            if i in dict.keys():
                items.append(dict[i][0])
        return items
    
    def split_data(item_index, item_list):
        tmp_list = []
        for i in sorted(item_index, reverse=True):
            tmp_list.append(item_list[i])
            del item_list[i]
        return tmp_list, item_list
    
    class_dict = get_class_dict(y)
    train_items = get_each_items(class_dict)
    train_x, x = split_data(train_items, _x)
    train_y, y = split_data(train_items, y)

    class_dict = get_class_dict(y)
    test_items = get_each_items(class_dict)
    test_x, x = split_data(test_items, x)
    test_y, y = split_data(test_items, y)
    
    class_dict = get_class_dict(y)
    valid_items = get_each_items(class_dict)
    valid_x, x = split_data(valid_items, x)
    valid_y, y = split_data(valid_items, y)
        
    random.seed(7777)
    rest_index = list(range(len(y)))
    random.shuffle(rest_index)
    
    valid_rest = test_size-len(valid_y)
    test_rest = test_size-len(test_y)
    valid_x = valid_x + [x[i] for i in rest_index[:valid_rest]]
    valid_y = valid_y + [y[i] for i in rest_index[:valid_rest]]
    test_x = test_x + [x[i] for i in rest_index[valid_rest:valid_rest+test_rest]]
    test_y = test_y + [y[i] for i in rest_index[valid_rest:valid_rest+test_rest]]
    train_x = train_x + [x[i] for i in rest_index[valid_rest+test_rest:]]
    train_y = train_y + [y[i] for i in rest_index[valid_rest+test_rest:]]
        
    assert len(_y) == len(train_y) + len(valid_y) + len(test_y)
    
    x_dict = {}
    y_dict = {}
    
    x_dict['train'] = train_x
    y_dict['train'] = train_y
    x_dict['valid'] = valid_x
    y_dict['valid'] = valid_y
    x_dict['test'] = test_x
    y_dict['test'] = test_y
        
    return x_dict, y_dict, y_class, class_dict

def get_unlabeled_data(processed_data, vocab, sp):
    x_U = []
    for sent in tqdm(processed_data):
        input_ids = np.array([vocab['[CLS]']]+ [vocab[t] for t in sp(sent)]+[vocab['[SEP]']])
        input_mask = np.ones(input_ids.shape, np.long)
        token_type_ids = np.zeros(input_ids.shape, np.long)
        x_U.append((input_ids, token_type_ids, input_mask))
    return x_U


class Dataset(Dataset):
    def __init__(self, x, y, contain_sent=False):
        self.X = x
        self.Y = y
        self.len = len(self.X)
        self.contain_sent = contain_sent # TBR
    def __getitem__(self, index):
        if self.contain_sent:
            input_ids, token_type_ids, input_mask, sent = self.X[index]
            return input_ids, token_type_ids, input_mask, sent, self.Y[index]
        else:
            input_ids, token_type_ids, input_mask, self.X[index]
            return input_ids, token_type_ids, input_mask, self.Y[index]
    def __len__(self):
        return self.len
    
    
def collate_fn(batch):
    try:
        _input_ids, _token_type_ids, _input_mask, _label = zip(*batch)
    except Exception:
        _input_ids, _token_type_ids, _input_mask, sent, _label = zip(*batch)
    
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

def get_data_dict(x_dict, y_dict, train=False, contain_sent=False):
    for t in ['train', 'valid', 'test']:
        x_dict['HEAD_'+t], x_dict['TAIL_'+t], y_dict['HEAD_'+t], y_dict['TAIL_'+t] = split_HEAD_TAIL(x_dict[t], y_dict[t])
    
    data_dict = {}
    loader_dict = {}
    for t1 in ['', 'HEAD', 'TAIL']:
        for t2 in ['train', 'valid', 'test']:
            key = t2
            if t1 != '':
                key = t1+'_'+key
            data_dict[key] = Dataset(x_dict[key], y_dict[key], contain_sent)
            if train and t2 == 'train':
                loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
            elif t2 != 'train':
                loader_dict[key] = DataLoader(dataset=data_dict[key], batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return x_dict, y_dict, data_dict, loader_dict
