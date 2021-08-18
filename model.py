from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from transformers import AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import model_utils
import numpy as np
import random
import os
import copy
import pickle

class BERTClassifier(nn.Module):
    def __init__(self, 
                 bert,
                 hidden_size = 768,
                 dr_rate=None,
                 params=None,
                 batch_size=16,
                 warmup_ratio=0.1,
                 num_of_epoch=200,
                 max_grad_norm=1,
                 learning_rate=0.00001,
                 criterion_name='CE',                 
                 num_of_classes=-1,
                 num_of_class_samples=None, # only for LDAM
                 device='cpu',
                 model_name='bert_classifier.ckpt',
                 random_seed=7777, 
                 cancel_pos=False,
                 additive=False):

        #super(CosSimModel, self).__init__()
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.cancel_pos = cancel_pos
        if self.cancel_pos:
            print('[Biased Model]')

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        random.seed(random_seed)
                 
        self.classifier = nn.Linear(hidden_size , num_of_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            
        self.warmup_ratio = warmup_ratio
        self.num_of_epoch = num_of_epoch
        self.batch_size = batch_size
        
        self.max_grad_norm = 1
        self.learning_rate = learning_rate
        self.criterion = model_utils.get_criterion(criterion_name, num_of_classes, num_of_class_samples=num_of_class_samples, device=device)
        
        for param in self.bert.parameters():
            param.requires_grad = True
        self.score = None
        self.name = model_name
        
        self.additive = additive
        if self.additive:
            print('[Load biased model]')
            self.prev_model = copy.deepcopy(self)
            self.prev_model.cancel_pos = True
            self.prev_model.additive=False
            #if not os.path.exists('./ckpt/cancel_pos/'+self.name) and os.path.exists('./ckpt/'+self.name):
            os.system('mv ./ckpt/'+self.name+' ./ckpt/cancel_pos/'+self.name)
            self.prev_model.load('cancel_pos/'+self.name)
            self.prev_model.eval()

    def get_repr(self, input_ids, token_type_ids, input_mask):
        pos_ids = None
        if self.cancel_pos:
            pos_ids = token_type_ids
        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, 
                              attention_mask=input_mask, position_ids=pos_ids)
        z = self.linear(pooled_output)
        return z

    def forward(self, input_ids, input_mask, token_type_ids):
        pos_ids = None
        if self.cancel_pos:
            pos_ids = token_type_ids
        _, pooler = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, 
                              attention_mask=input_mask, position_ids=pos_ids, return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)        
            
        return self.classifier(out)

    #def forward_with_output(self, input_ids, input_mask, token_type_ids):
    #    output, sequence_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, return_sequence_output=True)
    #    if self.dr_rate:
    #        out = self.dropout(output[1])
    #    return selier(out), sequence_output

    def _train(self, train_data, valid_data, test_data, save=True, always_eval_test=False):
        #self.load(name)
        self.train()
        self.bert.train()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}            
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        #print('leraning rate:', optimizer.param_groups[0]["lr"])
        t_total = len(train_data) * self.batch_size * self.num_of_epoch
        warmup_step = int(t_total * self.warmup_ratio)        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
        
        tmp_best_score = None
        for _ in range(self.num_of_epoch):
            total_loss = 0.0
            self.train()            
            print('\n[Epoch %s]' % _ )
            for i, batch in enumerate(train_data):
                optimizer.zero_grad()
                output = self.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids'])
                if self.additive:
                    prev_output = self.prev_model.forward(batch['input_ids'], 
                                                          batch['input_mask'], batch['token_type_ids']).detach()
                    output = output + prev_output
                
                loss = self.criterion(output, batch['label']).sum()
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                torch.cuda.empty_cache()
                if i % 40 == 0:
                    print('== Train loss at epoch %s and step %s: %s' % (str(_), str(i), str(round(total_loss / ((i + 1) * self.batch_size), 4))))
            
            self.eval()

            tmp_score, valid_balanced_score, valid_loss, report = self.evaluation(valid_data)            
            #print(report)
            
            save_condition = tmp_best_score is None or tmp_best_score <= valid_balanced_score
            if always_eval_test or save_condition:
                print('=== Validation loss, accuracy, balanced_accuracy at epoch %s: %s, %s, %s [SAVED]' % (str(_), str(round(valid_loss, 4)), round(tmp_score, 4), round(valid_balanced_score, 4)))
                tmp_best_score = valid_balanced_score                
                torch.cuda.empty_cache()
                test_score, test_balanced_score, test_loss, report = self.evaluation(test_data)
                print('==== Test loss, accuracy, balanced_accuracy at epoch %s: %s, %s, %s' % (str(_), str(round(test_loss, 4)), round(test_score, 4), round(test_balanced_score, 4)))
                #print('== Test accuracy, balanced_accuracy at epoch %s: %s, %s' % (str(_), round(tmp_score, 4), round(valid_balanced_score, 4)))
                #print('TEST:', round(test_score, 4), round(test_balanced_score, 4))
                #print(report)
                if save_condition:
                    self.score = test_balanced_score
                    self.save(self.name)
            else:
                print('=== Validation loss, accuracy, balanced_accuracy at epoch %s: %s, %s, %s [NOT SAVED]' % (str(_), str(round(valid_loss, 4)), round(tmp_score, 4), round(valid_balanced_score, 4)))
        self.eval()
        return tmp_best_score

    def evaluation(self, data, log=False):
        self.eval()
        self.bert.eval()
        pred = None
        
        if log:
            input_text = []
        total_loss = 0.0      
        count = 0
        for batch in data:            
            with torch.no_grad():
                output = self.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids'])
                loss = self.criterion(output, batch['label']).sum()
                p = torch.argmax(output, dim=1).cpu().numpy()

                total_loss += loss.item()
                if pred is None:
                    pred = p
                else:
                    pred = np.concatenate([pred, p], axis=0)
                if log:
                    input_text.append(batch['input_ids'].cpu().numpy())
                count += 1
        total_loss /= (count * self.batch_size)
        l = None
        for batch in data:
            _l = batch['label'].cpu().numpy()
            if l is None:
                l = _l
            else:
                l = np.concatenate([l, _l], axis=0)
                
        accuracy = (pred == l).astype(np.float32).sum() / l.shape[0]
        balanced_accuracy = balanced_accuracy_score(l.tolist(), pred.tolist())
        
        if log:
            with open('log_'+self.name, 'wb') as fp:
                pickle.dump((input_text, pred, l), fp, pickle.HIGHEST_PROTOCOL)
        
        return accuracy, balanced_accuracy, total_loss, classification_report(l, pred, digits=3)
    
    def save(self, name=None):
        if name is None:
            name = self.name
        torch.save({'state_dict': self.state_dict()}, 'ckpt/'+name)
 
    def load(self, name=None):
        if name is None:
            name = self.name
        if not os.path.exists('ckpt/'+name):
            print('No ckpt')
            return False
        ckpt = torch.load('ckpt/'+name, map_location='cuda:'+str(torch.cuda.current_device()))
        self.load_state_dict(ckpt['state_dict'], strict=False)
        print(name, 'loaded')
        return True
