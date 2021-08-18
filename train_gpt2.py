from config import *
from os import path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

#def choose_from_top(probs, n=5):
#    ind = np.argpartition(probs, -n)[-n:]
#    top_prob = probs[ind]
#    top_prob = top_prob / np.sum(top_prob) # Normalize
#    choice = np.random.choice(n, 1, p = top_prob)
#    token_id = ind[choice][0]
#    return int(token_id)

def save_result(result_filename, result_info_filename, ARGS, test_acc, test_balanced_acc, test_loss):
    result_id = 0
    if path.exists(result_filename):
        with open(result_filename, 'r') as fread:
            result_id = len(fread.readlines())
    
    fwrite = open(result_filename, 'a')
    fwrite_info = open(result_info_filename, 'a')

    fwrite.write('%s\t%s\t%s\n' % (result_id, round(test_acc, 4), round(test_balanced_acc, 4)))
    fwrite_info.write('== Test ID %s at %s ==\n' % (result_id, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    fwrite_info.write(str(ARGS) + '\n')
    fwrite_info.write('Test loss, accuracy, balanced_accuracy: %s, %s, %s\n\n' % (str(round(test_loss, 4)), round(test_acc, 4), round(test_balanced_acc, 4)) + '\n')    
    
    fwrite.close()
    fwrite_info.close()

class M2MDataset(Dataset):
    def __init__(self, filename, tokenizer, is_test=False, model_name='our'):
        super().__init__()       
        self.eos = '<|endoftext|>'
        
        self.X = []
        self.sent_set = set()
        
        fread = open(filename,'r')
        for line in fread.readlines():
            line = line.strip()            
            if line == '': continue                        
            
            #print(line)
            parsed_info_dict = self.parse_data_for_GPT(line, tokenizer, model_name, is_test)
            self.X.append(parsed_info_dict)
        
        print(np.shape(self.X))
        fread.close()
    
    def parse_data_for_GPT(self, line, tokenizer, model_name, is_test = False):
        row = dict()
        
        if model_name == 'our':
            splits = line.strip().split('\t')
            row['source_id'] = splits[0]            
            row['target_id'] = splits[1]                    
            row['source_sent'] = splits[2]            
            self.sent_set.add(row['source_sent'])
            
            source_id_token = '<|%s|>' % row['source_id']
            target_id_token = '<|%s|>' % row['target_id']           
            tokenizer.add_tokens([source_id_token, target_id_token])
                        
            input = source_id_token + tokenizer.sep_token + row['source_sent'] + tokenizer.sep_token + target_id_token + tokenizer.sep_token

            if is_test == False: # training case
                row['target_sent'] = splits[3]
                input += row['target_sent'] + self.eos
            
        elif model_name == 'lambada':
            # JW - RBR
            splits = line.strip().split('\t')#.replace('[class','').replace('] ','\t').split('\t')
            row['target_id'] = splits[0]
                        
            target_id_token = '<|%s|>' % row['target_id']           
            tokenizer.add_tokens([target_id_token])
            input = target_id_token + tokenizer.sep_token

            if is_test == False: # training case
                row['target_sent'] = splits[1]
                input += row['target_sent'] + self.eos

        else:
            raise (NameError('Allowing the following models', model_list))
        row['input_ids'] = tokenizer.encode(input)
        
        return row
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]
    
    def get_all_sent_set(self):
        return self.sent_set

def generate_sentence_gpu_topk(data, model, tokenizer, k):
    model.eval()
    print('Generation Number:', k)
    sent_list = sampling(data, model, tokenizer, k)
    return sent_list    

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def sampling(data, model, tokenizer, k):
    sent_list = list()
    input_ids = data
    temperature = 1.0
    top_p = 0.9
    top_k = 0
    max_repeat = 3
    
    with torch.no_grad():
        for i in range(k):
            target_sent = ''
            count = 0
            repeat = 0
            prev_gen = '<|endoftext|>'
            while count < MAX_SEQ_LEN:
                pred = model(input_ids)[0]
                logits = pred
                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, top_k)
                logits = top_p_logits(logits, top_p=top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                gen = tokenizer.decode(prev[0])
                if gen == '<|endoftext|>':
                    break
                
                if prev_gen == gen:
                    repeat += 1
                    if repeat == max_repeat: break                    
                else:
                    repeat = 0

                target_sent += gen
                input_ids = torch.cat((input_ids, prev), 1)
                count += 1
                prev_gen = gen
                #print('Target_sent:', target_sent)
            
            sent_list.append(target_sent.replace('▁', ' ').strip())
            input_ids = data
            #tokens = tokenizer.decode(input_ids.tolist()[0])
            #sent_set.add(tokens)
        
    return sent_list
    
if __name__ == '__main__':
    device = ARGS.device
    #if torch.cuda.is_available():
    #    device = 'cuda:0'    
    
    print('\n== Train GPT2 model ==')     

    # Hyper parameters
    EPOCHS = ARGS.num_of_epoch
    batch_size = ARGS.batch_size
    LEARNING_RATE = ARGS.learning_rate
    WARMUP_STEPS = 5000
    MAX_SEQ_LEN = 128
    save_path = './checkpoint_lambamda/'
    gmodel, model_name = 'lambada', 'lambada'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    proc_seq_count = 0
    avg_loss = (0.0, 0.0)
    count = 0

    tokenizer.add_tokens(['<|sep|>'])
    tokenizer.add_special_tokens({'sep_token': '<|sep|>'})

    train_file = './data/%s/m2m_train_%s' % (ARGS.dataset, ARGS.data_setting)
    train_dataset = M2MDataset(train_file, tokenizer, model_name=model_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.resize_token_embeddings(len(tokenizer)) 

    for epoch in range(1, EPOCHS + 1):
        for idx, row in enumerate(train_loader):
            model.train()
        
            optimizer.zero_grad()
            data = torch.stack(row['input_ids'])
            data = data.transpose(1,0)
            data = data.to(device)
        
            outputs = model(data, labels=data)        
            loss, logits = outputs[:2]  
            loss = loss.to(device)
            loss.backward()
            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
            optimizer.step()
            #scheduler.step()
        
            #if count % 10 == 0:
            print('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(epoch, count, loss, avg_loss[0] / avg_loss[1]))
            #summary.add_scalar('loss/avg_loss', avg_loss[0] / avg_loss[1], count)
            #summary.add_scalar('loss/loss', loss, count)
        
            count += 1

    print('[Sentence generation]')
    new_sentence_dict = defaultdict(list)
    new_sentence_meta_dict = defaultdict(list)
    total_gen = 0

    for target_cls in classes:    
        target_cls_counts = df_train_sent_group_by_label[df_train_sent_group_by_label['label'] == target_cls]['counts'].item()
        diff = max_count - target_cls_counts # min(max_count, 2 * target_cls_counts) - target_cls_counts
        print('Class: %s, Diff min(N1 - Ni): %s' % (str(target_cls), str(diff)))
        start_time = time.time()

        #while len(new_sentence_dict[target_cls]) < diff:
        #    if len(new_sentence_dict[target_cls]) % 100 == 0:            
                #print('%s-th sentence generation in class %s' % (len(new_sentence_dict[target_cls]), target_cls))

            # step1: source selection
        #data, source_id = row['input_ids'], row['target_id']
        #data = torch.stack(data)
        #data = data.transpose(1,0)
        data = torch.tensor(tokenizer.encode('<|%s|><|sep|>' % target_cls)).unsqueeze(0)
        data = data.to(device)
        target_sents = generate_sentence_gpu_topk(data, model, tokenizer, diff)
        new_sentence_dict[target_cls] = target_sents
            #new_sentence_meta_dict[target_cls].append(meta_dict)
            
        total_gen += len(target_sents)
        print(target_cls, diff, 'Done', str(time.time() - start_time))

    print('Total generation count:', total_gen)

    # Output
    filename = './data/' + ARGS.dataset + '/aug_%s_%s_%s_raw.csv' % (ARGS.data_setting, ARGS.cmodel, gmodel) # cmodel_name
    #filename_meta = './data/' + ARGS.dataset + '/aug_%s_%s_%s.meta' % (ARGS.data_setting, ARGS.cmodel, gmodel) # cmodel_name
    
    fwrite = open(filename, 'w')
    for cls, sentences in new_sentence_dict.items():
        for sentence in sentences:
            fwrite.write(str(cls) + '\t' + sentence + '\n')
    fwrite.close()

    #fwrite = open(filename_meta, 'w')
    #for cls, meta_dict_list in new_sentence_meta_dict.items():
    #    for meta_dict in meta_dict_list:
    #        fwrite.write(str(meta_dict) + '\n')
    #fwrite.close()    
    
