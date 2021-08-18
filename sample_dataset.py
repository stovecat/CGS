import pandas as pd
from utils_en import *
from model_utils import FocalLoss, LDAMLoss
import model_utils
from model import BERTClassifier
import os


def clear_seed_all(seed=7777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

# build dataset 
#   - containing dark knowledge
#   - balancing head (sampling) / tail

# note that trainset is non-shuffle
def build_sampled_dataset(teacher, x_dict, y_dict, loader_dict, data_dict, data_type, 
                          HEAD_labels, k_shot=None, imbalance=False, seed=7777):
    assert data_type in ['train', 'valid']
    def get_teacher_pred(teacher, loader):
        teacher.eval()
        teacher.bert.eval()
        pred = None

        with torch.no_grad():
            for batch in tqdm(loader):
                p = teacher.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids']).cpu().numpy()
                p = p[:, HEAD_labels]
                if pred is None:
                    pred = p
                else:
                    pred = np.concatenate([pred, p], axis=0)
        return pred

    teacher_pred = get_teacher_pred(teacher, loader_dict['HEAD_'+data_type])
    
    def get_indexes(data):
        random.seed(seed)
        dict = {}
        for i, y in enumerate(data.Y):
            y = int(y)
            if y not in dict.keys():
                dict[y] = []
            dict[y].append(i)
        for key in dict.keys():
            random.shuffle(dict[key])
#         for y in dict.keys():
#             print(y_class[y], len(dict[y]))
        return dict

    def make_k_shot(data, k_shot):
        for key in data.keys():
            data[key] = data[key][:k_shot]
        return data

    H_dict = get_indexes(data_dict['HEAD_'+data_type])
    T_dict = get_indexes(data_dict['TAIL_'+data_type])
    if k_shot is not None:
        T_dict = make_k_shot(T_dict, k_shot)
    
    TAIL_labels = list(T_dict.keys())
    
    set_key = 'sampled_'+data_type
    
    if imbalance:
        avg_tail_data = len(data_dict['HEAD_'+data_type])
    else:
        if k_shot is None:
            avg_tail_data = int(sum([len(T_dict[y]) for y in T_dict.keys()])/len(list(T_dict.keys())))
        else:
            avg_tail_data = k_shot
    x_dict[set_key] = []
    y_dict[set_key] = []
    d_dict = {}
    d_dict[set_key] = []
    pseudo_distill = np.array([-1.]*len(HEAD_labels))
    for key in H_dict.keys():
        x_dict[set_key].extend([x_dict['HEAD_'+data_type][i] for i in H_dict[key][:avg_tail_data]])
        y_dict[set_key].extend([y_dict['HEAD_'+data_type][i] for i in H_dict[key][:avg_tail_data]])
        d_dict[set_key].extend([teacher_pred[i] for i in H_dict[key][:avg_tail_data]])
    for d in d_dict[set_key]: #pseudo distill validation check
        if sum([d[i] == pseudo_distill[i] for i in range(len(HEAD_labels))]) == len(HEAD_labels):
            assert 1 == 2
    for key in T_dict.keys():
        x_dict[set_key].extend([x_dict['TAIL_'+data_type][i] for i in T_dict[key]])
        y_dict[set_key].extend([y_dict['TAIL_'+data_type][i] for i in T_dict[key]])
        d_dict[set_key].extend([pseudo_distill]*len(T_dict[key]))
        
    class DistillDataset(Dataset):
        def __init__(self, x, y, d):
            self.X = x
            self.Y = y
            self.D = d
            self.len = len(self.X)
        def __getitem__(self, index):
            input_ids, token_type_ids, input_mask = self.X[index]
            #print('Y', self.Y[index])
            return input_ids, token_type_ids, input_mask, self.Y[index], self.D[index]
        def __len__(self):
            return self.len
        
    def collate_fn(batch):
        _input_ids, _input_mask, _token_type_ids, _label, _distill = zip(*batch)

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
        distill = np.zeros([bs, len(HEAD_labels)], np.float32)


        for i in range(bs):
            for j in range(len(_input_ids[i])):
                if j >= max_len:
                    break
                input_ids[i, j] = _input_ids[i][j]
                input_mask[i, j] = _input_mask[i][j]
                token_type_ids[i, j] = _token_type_ids[i][j]
                if j < len(HEAD_labels):
                    distill[i, j] = _distill[i][j]
            #label[i, _label[i]] = 1
            label[i] = _label[i]

        batch = {}
        batch['input_ids'] = input_ids
        batch['input_mask'] = input_mask
        batch['token_type_ids'] = token_type_ids
        batch['label'] = np.concatenate([label.reshape(bs, 1), distill], axis=1)
        #batch['distill'] = distill

        for key in batch.keys():
            batch[key] = torch.tensor(np.asarray(batch[key]))
            if gpu:
                batch[key] = batch[key].cuda()    
        return batch
        
    data_dict[set_key] = DistillDataset(x_dict[set_key], y_dict[set_key], d_dict[set_key])
    loader_dict[set_key] = DataLoader(dataset=data_dict[set_key], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)#False)#
        
    return x_dict, y_dict, d_dict, data_dict, loader_dict, TAIL_labels


class Student():
    def __init__(self, 
                 model,
                 criterion_name='CE',                 
                 num_of_classes=-1,
                 num_of_class_samples=None, # only for LDAM
                 device='cpu',
                 model_name='bert_classifier.ckpt',
                 distillation=False,
                 distillation_option=None,
                 num_of_classes_tail=-1,
                 num_of_class_samples_tail=None,
                 HEAD_labels=None, 
                 seed=7777
                ):
        #super(Student, self).__init__()
        
        clear_seed_all(seed)
        self.distillation = distillation
        self.distillation_option = distillation_option
        self._criterion = model_utils.get_criterion(criterion_name, num_of_classes, num_of_class_samples=num_of_class_samples, device=device)
        self._criterion_tail = model_utils.get_criterion(criterion_name, num_of_classes_tail, num_of_class_samples=num_of_class_samples_tail, device=device)
        self.HEAD_labels = HEAD_labels
        self.NON_HEAD_labels = [i for i in range(num_of_classes) if i not in HEAD_labels]
        self.model = model
        del self.model.criterion
        self.model.criterion = self.criterion        
        self.model.name = model_name
        
    def train(self, loader_dict, dataset_keys=['sampled_train', 'sampled_valid', 'test'], always_eval_test=False):
        return self.model._train(loader_dict['sampled_train'], loader_dict['valid'], 
                                   loader_dict['test'], always_eval_test=always_eval_test)
        
    def eval(self, loader):
        test_score, test_balanced_score, test_loss, report = self.model.evaluation(loader)
        print(test_score, test_balanced_score)
        print(report)
    
    def load(self, path=None):
        self.model.load(path)
        
    def cuda(self, dev):
        self.model.cuda(dev)
        
    def to(self, dev):
        self.model.to(dev)
        
    def criterion(self, output, _label):
        if len(_label.size()) == 1: #dataloader without distillation
            return self._criterion(output, _label)
        
        label, distill = torch.round(_label[:, 0]).long(), _label[:, 1:]
        if not self.distillation: #no distillation
            return self._criterion(output, label)
        
        
        head_idx = None
        for h in self.HEAD_labels:
            index = (label == h).nonzero().view(-1).long()
            if head_idx is None:
                head_idx = index
            else:
                head_idx = torch.cat([head_idx, index], dim=0)
        tail_idx = torch.tensor([index for index in range(label.size()[0]) if index not in head_idx]).long()
        
        #HEAD labels
        head_o = output.index_select(0, head_idx)
        head_l = label.index_select(0, head_idx)
        head_d = distill.index_select(0, head_idx)
        if head_o.size()[0] == 0:
            head_loss = torch.tensor(0.)
        else:
            head_o_head_logits = head_o.index_select(1, torch.tensor(self.HEAD_labels))
            tail_labels = torch.tensor([HT2T(l, self.HEAD_labels) for l in head_l])
            head_o_tail_logits = head_o.index_select(1, torch.tensor(self.NON_HEAD_labels))
            if self.distillation_option == 'L2_logits_default_tail':
                assert 1 == 2
                #distillation loss
                gap = head_d - head_o_head_logits
                distill_loss = (gap * gap).mean(dim=1)
                
                #teacher loss except head logits
                head_o_tail_loss = self._criterion_tail(head_o_tail_logits, tail_labels)

                head_loss = (distill_loss + head_o_tail_loss).mean()
            elif self.distillation_option == 'L2_logits_L2_min_tail':
                #head distillation loss
                gap_h = head_d - head_o_head_logits
                
                #head distillation loss
                min_logits = output.min()
                tail_d = min_logits.reshape(1, 1).repeat(head_o_tail_logits.size())
                gap_t = tail_d - head_o_tail_logits
                
                gap = torch.cat([gap_h.double(), gap_t.double()], dim=1)
                distill_loss = (gap * gap).mean(dim=1)
                head_loss = distill_loss.mean()
            elif self.distillation_option == 'D+FullCE':
                #head distillation loss
                gap_h = head_d - head_o_head_logits
                
                #head distillation loss
                min_logits = output.min()
                tail_d = min_logits.reshape(1, 1).repeat(head_o_tail_logits.size())
                gap_t = tail_d - head_o_tail_logits
                
                gap = torch.cat([gap_h.double(), gap_t.double()], dim=1)
                distill_loss = (gap * gap).mean(dim=1)
                head_loss = distill_loss.mean()
            else:
                raise NotImplementedError
        
        #TAIL labels
        tail_o = output.index_select(0, tail_idx)
        tail_l = label.index_select(0, tail_idx)
        if tail_o.size()[0] == 0:
            tail_loss = torch.tensor(0.)
        else:
            #teacher loss
            tail_loss = self._criterion(tail_o, tail_l).mean()
            
        if self.distillation_option == 'D+FullCE':
            full_loss = self._criterion(output, label).mean()
            return (full_loss + head_loss + tail_loss).reshape(1, 1)

            
        return (head_loss + tail_loss).reshape(1, 1)

    
def HT2T(index, HEAD_labels):
    offset = 0
    for h in sorted(HEAD_labels):
        if index > h:
            offset += 1
    return index-offset

def get_num_of_class_samples(data_dict, class_labels, HEAD_labels, opt='tail'):
    assert opt in ['head', 'tail', 'all']
    sents, labels = [], []
    for data in data_dict:
        sents.append(data[0])
        labels.append(int(data[3]))
    
    df_sent = pd.DataFrame(list(zip(sents, labels)), columns=['sent', 'label'])
    num_of_class_samples = [0] * len(class_labels)

    df_label = df_sent.groupby(['label']).size().reset_index(name='counts')
    max_samples = df_label['counts']
    for index, row in df_label.iterrows():
        _label = row.label
        if opt == 'tail':
            _label = HT2T(_label, HEAD_labels)
        elif opt == 'all':
            pass
        elif opt == 'head':
            raise NotImplementedError
        num_of_class_samples[_label] = row.counts
    return num_of_class_samples

