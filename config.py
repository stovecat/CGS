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

from utils_en import *
from model_utils import FocalLoss, LDAMLoss


def parse_args():
    parser = argparse.ArgumentParser(description='TranGen Training')
    parser.add_argument('--gpu', action='store_true', help='use of GPU') 
    parser.add_argument('--cpu', dest='gpu', action='store_false', help='use of GPU') 
    parser.set_defaults(gpu=False)

    parser.add_argument('--device', default=0, type=int, help='GPU device') 
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_of_epoch', default=200, type=int, help='total epochs to run')
    parser.add_argument('--random_seed', default=7777, type=int, help='random seed')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='warmup-ratio')
    parser.add_argument('--max_grad_norm', default=1., type=float, help='max grad norm')
    parser.add_argument('--MAX_LEN', default=128, type=int, help='max length of words in one sentence')
    parser.add_argument('--dr_rate', default=0.5, type=float, help='drop-out ratio of a classfication layer')

    parser.add_argument('--dataset', default=None, type=str, choices=['ATIS', 'TREC', 'SNIPS', 'SENT'], help='dataset')
    parser.add_argument('--data_setting', default=None, type=str, choices=['all', 'longtail', 'step'], help='data setting')    
    parser.add_argument('--imbalanced_ratio', default=None, type=int, choices=[10, 100], help='imbalanced ratio')

    parser.add_argument('--train_bert', action='store_true', help='a flag of update/no update of parameters of BERT') # CPU?
    parser.add_argument('--no_train_bert', dest='train_bert', action='store_false', help='a flag of update/no update of parameters of BERT') # CPU?
    parser.set_defaults(train_bert=True)    
    parser.add_argument('--loss_type', default='CE', type=str, choices=['CE', 'Focal', 'LDAM'], help='a loss type')
    
    parser.add_argument('--data_augment', action='store_true', help='a flag of data augmentation for training the classifier') # CPU?    
    parser.set_defaults(data_augment=False)    

    parser.add_argument('--cmodel', default=None, type=str, choices=['our', 'standard', 'Focal'], help='a classification model') # our: LDAM, standard: CE
    parser.add_argument('--gmodel', default=None, type=str, choices=['bart', 'our', 'lambada'], help='a generation model')

    parser.add_argument('--result_path', default='./result/', type=str)        
    parser.add_argument('--min_valid_data_num', default=5, type=int) 

    # for translation
    parser.add_argument('--beta', default=0.99, type=float, help='beta')
    parser.add_argument('--mask_ratio', default=0.2, type=float, help='mask_ratio of bart_span')
    parser.add_argument('--target_sample_num', default=None, choices=['N1', 'Nk'], type=str, help='for bart')
    parser.add_argument('--source_selection', default='random', choices=['random', 'cluster'], type=str, help='for bart')
    parser.add_argument('--use_token_importance_file', dest='use_token_importance_file', action='store_true', help='use token importance') # CPU?
    parser.set_defaults(use_token_importance_file=False) 
    
    # for HEAD (temporal)
    parser.add_argument('--head_only', action='store_true', help='HEAD data only') 
    parser.set_defaults(head_only=False)    
    
    parser.add_argument('--teacher', default=None, type=str, help='a teacher model name for WARM_LEARNING')
    parser.add_argument('--cancel_pos', default=False, type=bool)
    parser.add_argument('--additive', default=False, type=bool)
    parser.add_argument('--binary_class_idx', default=None, type=int)
    parser.add_argument('--eval_only', default=False, type=bool)
    parser.add_argument('--eval_log', default=False, type=bool)
    
    parser.add_argument('--TMix', default=False, type=bool)
    parser.add_argument('--HEAD_labels', default=None, type=int)
    parser.add_argument('--TAIL_labels', default=None, type=int)
    parser.add_argument('--TOTAL_labels', default=None, type=int)

    
    return parser.parse_args()

ARGS = parse_args()
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
elif ARGS.dataset == 'SENT':
    N_SAMPLES = 1707
    #N_SAMPLES = 100
else:
    print('Dataset:', ARGS.dataset)
    raise NotImplementedError()

random.seed(ARGS.random_seed)
np.random.seed(ARGS.random_seed)
torch.manual_seed(ARGS.random_seed)
torch.cuda.manual_seed_all(ARGS.random_seed)

# To be refactored
model_name = '%s_%s_%s_classifier_%s_%s_%s_%s.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), ARGS.loss_type, ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel)

if ARGS.TMix:
    from model_tmix import get_bert, MixText as BERTClassifier
    bert, tokenizer = get_bert()
    print('\n== Use TMix ==')
else:
    from model import BERTClassifier
    bert, tokenizer = get_bert()
if ARGS.gpu:
    bert.cuda(ARGS.device)
else:
    bert.cpu()


    
print('\n== Load data ==')
labeled_train_data, labeled_valid_data, labeled_test_data, HEAD_labels, TAIL_labels = load_data(ARGS.dataset, data_setting=ARGS.data_setting, imbalanced_ratio=ARGS.imbalanced_ratio, head_sample_num=N_SAMPLES, min_valid_data_num=ARGS.min_valid_data_num)
ARGS.HEAD_labels = HEAD_labels
ARGS.TAIL_labels = TAIL_labels



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

if ARGS.binary_class_idx is not None:
    model_name = '%s_%s_%s_classifier_%s_%s_%s_%s_%s_%s.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), ARGS.loss_type, ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel, ARGS.binary_class_idx, ARGS.random_seed)
    labeled_train_data, labeled_train_data_rest = convert2binary(labeled_train_data, ARGS.binary_class_idx, 
                                                                 train=True, seed=ARGS.random_seed, aug=ARGS.data_augment)
    labeled_valid_data = convert2binary(labeled_valid_data, ARGS.binary_class_idx, train=False, 
                                        seed=ARGS.random_seed, aug=ARGS.data_augment)
    labeled_test_data = convert2binary(labeled_test_data, ARGS.binary_class_idx, train=False, 
                                       seed=ARGS.random_seed, aug=ARGS.data_augment)
    
x_dict, y_dict, y_class, class_dict = convert_data(labeled_train_data, labeled_valid_data, labeled_test_data, tokenizer)
num_of_classes = len(y_class)
x_dict, y_dict, data_dict, loader_dict = get_data_dict(x_dict, y_dict, train=True, shuffle=True, skip=True, nclasses=num_of_classes, balanced_sampling=ARGS.TMix)

df_train_data_filtered = labeled_train_data[['sentence', 'label']]
df_train_sent_group_by_label = df_train_data_filtered.groupby(['label']).size().reset_index(name='counts')
classes, class_counts = np.unique(df_train_sent_group_by_label['label'].tolist(), return_counts=True)



num_of_class_samples = [0] * (max(y_class) + 1)
max_samples = df_train_sent_group_by_label['counts']
max_count = max(df_train_sent_group_by_label['counts'].tolist())
ARGS.TOTAL_labels = num_of_classes


for index, row in df_train_sent_group_by_label.iterrows():        
    num_of_class_samples[row.label] = row.counts    

if 'translation.py' in sys.argv[0] or 'data_generation_for_gmodel.py' in sys.argv[0] or 'translation' in sys.argv[0]:
    from fairseq.models.bart import BARTModel
    from transformers import BartTokenizer
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
elif 'token_importance.py' in sys.argv[0] == False:
    print('Tokenizer is deleted!')
    del tokenizer

torch.cuda.empty_cache()
