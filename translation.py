# --dataset TREC --data_setting step --loss_type CE --imbalanced_ratio 100 --batch_size=16

from config import *
from fairseq.models.bart import BARTModel
from transformers import BartTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import Counter

def get_class_selection_weights(df_train_sent_group_by_label, beta):
    class_selection_weight_dict = defaultdict(dict)
    
    # calculation of class sepection probs for each class
    for i in range(len(classes)):
        class1 = classes[i]
        class1_counts = df_train_sent_group_by_label[df_train_sent_group_by_label['label'] == class1]['counts'].item()
        
        for j in range(i + 1, len(classes)):
            class2 = classes[j]
            class2_counts = df_train_sent_group_by_label[df_train_sent_group_by_label['label'] == class2]['counts'].item()

            diff = class2_counts - class1_counts
            if diff > 0:
                prob = math.pow(beta, diff)
                class_selection_weight_dict[class1][class2] = 1 - prob
                class_selection_weight_dict[class2][class1] = 0
            else:
                prob = math.pow(beta, -diff)
                class_selection_weight_dict[class1][class2] = 0
                class_selection_weight_dict[class2][class1] = 1 - prob

    # normalization for each calss    
    class_selection_dict = defaultdict(dict)
    for cls in classes:
        target_classes = list(class_selection_weight_dict[cls].keys())
        target_probs = list(class_selection_weight_dict[cls].values())        
        total_probs = sum(target_probs)
        
        if total_probs != 0:
            normalized_target_probs = [target_probs[i] / sum(target_probs) for i in range(len(target_probs))]                        
        else:
            normalized_target_probs = target_probs
            
        class_selection_dict[cls]['classes'] = list(target_classes)
        class_selection_dict[cls]['probs'] = list(normalized_target_probs)
    
    return class_selection_dict


def get_source_token_importances(cmodel, source_element, target_cls):
    input_ids = torch.tensor(source_element['input_ids'].item()).unsqueeze(0)
    input_mask = torch.tensor(source_element['input_mask'].item()).unsqueeze(0)
    token_type_ids = torch.tensor(source_element['token_type_ids'].item()).unsqueeze(0)
    label = torch.tensor(source_element['label'].item()).unsqueeze(0)        
    sentence = source_element['sentence'].item()

    tokenized_text = []
    attention_weights = []

    for token_id in input_ids[-1][1:]:
        if tokenizer.sep_token_id == token_id: break
        token = tokenizer.decode(int(token_id))
        tokenized_text.append(token.replace(' ', ''))

    output = cmodel.forward(input_ids, input_mask, token_type_ids)        
    loss = cmodel.criterion(output, torch.tensor(target_cls).unsqueeze(0))
    cmodel.zero_grad()
    loss.backward()
    #torch.cuda.empty_cache()

    importances = torch.tensor([])
    total_importances = 0.0

    for token_index in range(1, len(input_ids[-1])):
        token_id = input_ids[-1][token_index]
        if tokenizer.sep_token_id == token_id: break                

        importance = torch.norm(cmodel.bert.embeddings.position_embeddings.weight.grad[token_index], 2)
        total_importances += importance
        importances = torch.cat((importances , importance.unsqueeze(0)), dim=-1)
    
    return tokenized_text, importances

def merge_subwords(token_list, score_list, sub_word_merge_type):
    assert len(token_list) == len(score_list)
    merged_token_list, merged_score_list, merged_token_count_list = list(), list(), list()
    
    for i in range(len(token_list)): # token-level iteration 
        if i > 0 and token_list[i].startswith('##'):
            merged_token_list[-1] += token_list[i][2:]
            merged_token_count_list[-1] += 1

            if sub_word_merge_type == 'mean':                        
                merged_score_list[-1] = (((merged_token_count_list[-1] - 1) * merged_score_list[-1] + score_list[i])) / merged_token_count_list[-1]
            elif sub_word_merge_type == 'max':
                merged_score_list[-1] = max(merged_score_list[-1], score_list[i])
        else:
            merged_token_list.append(token_list[i])
            merged_token_count_list.append(1)            
            merged_score_list.append(score_list[i])
    
    assert len(merged_token_list) == len(merged_score_list)
    return merged_token_list, merged_score_list

def select_source_class(class_selection_dict, cls):    
    return np.random.choice(class_selection_dict[cls]['classes'], p=class_selection_dict[cls]['probs'])

def select_source_sentence(df_train, cls, type='class'):
    if type == 'class':
        return df_train[df_train['label'] == cls].sample()
    elif type == 'cluster':
        return df_train[df_train['cluster_id'] == cls].sample()
    else:
        raise NotImplementedError()

def get_cluster_source_sequeces(df_train, cls, type='class'):
    if type == 'class':
        return df_train[df_train['label'] == cls].sample()
    elif type == 'cluster':
        return df_train[df_train['cluster_id'] == cls].sample()
    else:
        raise NotImplementedError()

def masking_words(words): # 추후 곱연산으로 변경 가능
    for i in range(len(words)):
        if random.uniform(0, 1) < 0.2:
            words[i] = '<mask>'

def convert_words_to_sentence(words): # with merging masks
    '''
    Input: a list of words (masked)
    Output: a pair of <s1, s2> 
            - s1 is a string of words (adjacent masks are merged)
            - s2 is a string of words (adjacent masks are merged)
    '''
    prev_mask = False
    new_sentence_all_masks = ''
    new_sentence = ''

    for i in range(len(words)):
        is_mask_word = words[i] == '<mask>'
        new_sentence_all_masks += ' ' + words[i]
        if prev_mask and is_mask_word:
            continue
        else:
            new_sentence += ' ' + words[i]
            prev_mask = is_mask_word
    
    return new_sentence.strip(), new_sentence_all_masks.strip()

def bart_generation(element, sampling=True):
    sentence = element['sentence'].item()
    sentence_ids = bart_tokenizer.encode(sentence.strip(), add_special_tokens=False, padding=True, truncation=True, return_tensors='pt').to(gmodel_device)#.squeeze().to(gmodel_device)
    sentence_ids = sentence_ids.squeeze()
    source = torch.cat((bart_tokenizer.bos_token_tensor, bart_tokenizer.class_token_tensors[target_cls], sentence_ids, bart_tokenizer.eos_token_tensor)) # GPU
    source = add_whole_word_mask(source, ARGS.mask_ratio) # masking
    assert source[0] == 0 and source[-1] == 2 and source[1] != bart_tokenizer.mask_idx    
    
    gmodel_input_ids = source.unsqueeze(0).to('cpu')#.to('cpu')#.to(gmodel_device)
    if sampling == True:
        generated_output = gmodel.generate(gmodel_input_ids, sampling=True, sampling_topk=0, sampling_topp=0.9, beam=1, max_len_b=256, no_repeat_ngram_size=3)
    else:
        generated_output = gmodel.generate(gmodel_input_ids, beam=5, max_len_b=256, no_repeat_ngram_size=3)#.to(gmodel_device)

    generated_ids = generated_output[0]['tokens'].to('cpu')#.to(gmodel_device)#.to('cpu')
    rightmost_zero_index = np.argwhere(generated_ids== 0)[-1][-1]
    generated_str = bart_tokenizer.decode(generated_ids[rightmost_zero_index + 2:-1])

    meta_dict = dict()
    meta_dict['iteration (# of mask)'] = -1
    meta_dict['model'] = 'bart'
    meta_dict['label'] = element['label'].item()
    meta_dict['sentence'] = element['sentence'].item()            
    meta_dict['generated_sentence'] = generated_str
    
    return generated_str, meta_dict

def is_beginning_of_word(x):
    return bart_tokenizer.decode(x).startswith(' ')

def add_whole_word_mask(source, p):
    is_word_start = bart_tokenizer.mask_whole_words.gather(0, source) 

    is_word_start[0] = 0
    is_word_start[1] = 0
    is_word_start[2] = 1
    is_word_start[-1] = 0
    #print(is_word_start)

    num_to_mask = int(math.ceil(is_word_start.float().sum() * p))

    if num_to_mask == 0:
        return source
    
    #if self.mask_span_distribution is not None:
    lengths = torch.ones((num_to_mask,)).long()
    assert is_word_start[-1] == 0

    word_starts = is_word_start.nonzero()
    indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)    
    
    source_length = len(source)
    assert source_length - 1 not in indices    

    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
    source[indices] = bart_tokenizer.mask_idx
    
    while indices.size(0) > 0:
        next_masked = source[indices + 1] == bart_tokenizer.mask_idx
        indices_next_masked = indices[next_masked] + 1
        to_keep[indices_next_masked] = 0

        uncompleted = is_word_start[indices + 1] == 0
        indices = indices[uncompleted] + 1
        to_keep[indices] = 0

    source = source[to_keep]
    return source

def batch_merge_subwords(token_list, score_list, sub_word_merge_type):
    words_list, word_importances_list = [], []

    for tokens, token_importances in zip(token_list, score_list):
        word_importances_for_labels = []        
        # TBR (batch processing)
        for i in range(len(token_importances)): 
            words, word_importances = merge_subwords(tokens, token_importances[i], sub_word_merge_type)            
            word_importances_for_labels.append(word_importances)
        words_list.append(words)
        word_importances_list.append(word_importances_for_labels)
    return words_list, word_importances_list 

def read_token_importance(filename):    
    label_list, sent_list, token_list, score_list = [], [], [], []
    fread = open(filename, 'r', encoding='utf-8')

    for index, line in enumerate(fread.readlines()):
        splits = line.strip().split('\t')        
    
        label = splits[0]
        #label_text = idx_to_label_text_dict[int(label)]
        
        sentence = splits[1]
        label_list.append(int(label))
        
        #sent_list.append(('[class %s:%s] %s'% (label, label_text, sentence), index))
    
        token_list.append(list())
        score_list.append(list())
    
        for token in sentence.split(' '):
            token_list[-1].append(token)
    
        for scores in splits[2:]:
            score_list[-1].append(list())
            for score in scores.split(' '):
                score_list[-1][-1].append(float(score))

        while ' ##' in sentence: # 나중에는 원본 문장을 파일에 출력하여서 이용하는게 좋을 듯
            sentence = sentence.replace(' ##', '')
        
        sent_list.append(sentence)

    return label_list, sent_list, token_list, score_list

def gmodel_input_ids(row):    
    ## source
    sentence = row['sentence']
    sentence_ids = bart_tokenizer.encode(sentence.strip(), add_special_tokens=False, padding=True, truncation=True, return_tensors='pt').to(gmodel_device)#.squeeze().to(gmodel_device)
    sentence_ids = sentence_ids.squeeze()
    #source = torch.cat((bart_tokenizer.bos_token_tensor, bart_tokenizer.class_token_tensors[target_cls], sentence_ids, bart_tokenizer.eos_token_tensor)) # GPU
    source = torch.cat((bart_tokenizer.bos_token_tensor, bart_tokenizer.bos_token_tensor, sentence_ids, bart_tokenizer.eos_token_tensor)) # GPU
    return source

def get_is_word_start(row):
    source = row['source']
    
    is_word_start = bart_tokenizer.mask_whole_words.gather(0, source) 
    is_word_start[0] = 0
    is_word_start[1] = 0
    is_word_start[2] = 1
    is_word_start[-1] = 0

    assert is_word_start[-1] == 0
    is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
    return is_word_start

# two tokenizer are not matched
def get_word_to_bart_token_aligns(row):
    source = row['source']
    is_word_start = row['is_word_start']
    words = row['words']
    word_to_bart_token_aligns = []    

    tokens = []
    for i in range(len(source)):
        token = bart_tokenizer.decode([source[i]])
        tokens.append(token)
            
    token_index = 1
    token = ''
    for i, word in enumerate(words):
        is_first_tokens = True
        word_to_bart_token_aligns.append([])

        while len(word) > 0:
            if len(token) == 0:
                token_index += 1
                token = tokens[token_index]
                if token[0] == ' ': 
                    token = token[1:]
                if is_first_tokens is False:
                    is_word_start[token_index] = 0
            
            if word.startswith(token):
                word = word[len(token):]
                is_first_tokens = False
                word_to_bart_token_aligns[-1].append(token_index)
                token = ''
            elif len(token) == 1 and token == '�': # special characters1
                word = word[1:]
                is_first_tokens = False
                word_to_bart_token_aligns[-1].append(token_index)
                token_index += 1
                word_to_bart_token_aligns[-1].append(token_index)                
                token = ''
            else:
                raise NotImplementedError()
    #for i in range(2, len(source) - 1):
    #    token = bart_tokenizer.decode([source[i]])        
    #    token = token.replace(' ')

    #    #if is_word_start[i] == 1:
    #    #    if (i == 2 and words[word_index].startswith(token)) or words[word_index].startswith(token[1:]):
    #    #        word_index += 1
    #    #    else:
    #    #        raise NotImplementedError()
    #    #else:
    #    #    if words[word_index].startswith(token):
    #    #        is_word_start[i] = 1    
    #    #        word_index += 1

    #for i in range(2, len(source) - 1):
    #    if is_word_start[i] == 1:
    #        word_to_bart_token_aligns.append([i])
    #    else:
    #        word_to_bart_token_aligns[-1].append(i)

    assert len(word_to_bart_token_aligns) == len(words) and token_index == len(source) - 2 and token == '' #and len(words) == word_index
    return word_to_bart_token_aligns

if __name__ == '__main__':
    # Needs to move parameters to the config file
    sub_word_merge_type = 'mean'
    cmodel_device = 0
    gmodel_device = 0

    # INIT
    #bart_tokenizer.to(gmodel_device)
    bart_tokenizer.mask_idx = 50264 
    bart_tokenizer.eos_token_tensor = bart_tokenizer.encode([bart_tokenizer.eos_token], add_special_tokens=False, return_tensors='pt').squeeze(0).to(gmodel_device)
    bart_tokenizer.bos_token_tensor = bart_tokenizer.encode([bart_tokenizer.bos_token], add_special_tokens=False, return_tensors='pt').squeeze(0).to(gmodel_device)
    bart_tokenizer.mask_whole_words = torch.ByteTensor(list(map(is_beginning_of_word, range(bart_tokenizer.vocab_size)))).to(gmodel_device)
    bart_tokenizer.class_token_tensors = []
    
    cmodel_loss_type = ''    
    if ARGS.cmodel == 'our': 
        cmodel_loss_type = 'LDAM'    
    elif ARGS.cmodel == 'standard': 
        cmodel_loss_type = 'CE'    
    elif ARGS.cmodel == 'Focal': 
        cmodel_loss_type = 'Focal'    
    #cmodel_name = '%s_%s_%s_classifier_%s_%s_False_None.ckpt' % (ARGS.dataset, ARGS.data_setting, cmodel_loss_type, str(ARGS.learning_rate))
    cmodel_name = '%s_%s_%s_classifier_%s_%s_False_None.ckpt' % (ARGS.dataset, ARGS.data_setting, ARGS.imbalanced_ratio, cmodel_loss_type, str(ARGS.learning_rate))

    gmodel_checkpoint_path = './revised_libs/fairseq/checkpoints/'

    if ARGS.gmodel == 'our':
        gmodel_checkpoint_name = 'checkpoint20_%s_%s_%s.pt' % (ARGS.dataset, ARGS.data_setting, ARGS.gmodel) # need to revise ckpt name
    elif ARGS.gmodel == 'bart':
        gmodel_checkpoint_name = 'checkpoint_best_%s_%s_%s.pt' % (ARGS.dataset, ARGS.data_setting, ARGS.gmodel) # need to revise ckpt name
    elif ARGS.gmodel == 'lambada': # TBR
        augment_data_filename = './data/%s/aug_%s_None_%s_raw.csv' % (ARGS.dataset, ARGS.data_setting, ARGS.gmodel)
        labeled_aug_data = augment_data(augment_data_filename, tokenizer)
        x_dict, y_dict, y_class, class_dict = convert_data(labeled_aug_data, labeled_valid_data, labeled_test_data, tokenizer)
        print(x_dict.keys())
        for key, value in x_dict.items():
            print(key, len(value))
        #sys.exit(0)
        x_dict, y_dict, data_dict, loader_dict = get_data_dict(x_dict, y_dict, train=True, shuffle=True, skip=True)
    else:
        raise NotImplementedError()

    #gmodel_checkpoint_name = 'checkpoint_last_%s_%s_%s.pt' % (ARGS.dataset, ARGS.data_setting, ARGS.gmodel) # need to revise ckpt name

    reject_count = 0
    min_mask_treshold = 0.0
    new_sentence_dict = defaultdict(list)
    new_sentence_meta_dict = defaultdict(list)

    imb_ratio = 5
    total_gen, our_gen, bart_gen = 0, 0, 0    

    print('\n== Load Classifier Model: %s ==' % (cmodel_name))
    if ARGS.gmodel == 'our' or ARGS.gmodel == 'lambada':
        cmodel = BERTClassifier(bert, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name=cmodel_loss_type, num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=cmodel_name)
        cmodel.name = cmodel_name
        cmodel.to(cmodel_device)
        model_load_result = cmodel.load()
        assert model_load_result == True
        cmodel.eval()
    else: # BART does not need cmodel
        assert ARGS.cmodel == None
        print('Do not need to load cmodel for gmodel %s' % ARGS.gmodel)

    print('\n== Load Generator Model ==')
    if ARGS.gmodel == 'our' or ARGS.gmodel == 'bart':
        print('PATH:', gmodel_checkpoint_path+gmodel_checkpoint_name)
        gmodel = BARTModel.from_pretrained(gmodel_checkpoint_path, checkpoint_file=gmodel_checkpoint_name).to(gmodel_device)#, map_location='cpu')
        #gmodel = torch.hub.load('pytorch/fairseq', 'bart.large') #BARTModel.from_pretrained(gmodel_checkpoint_path, checkpoint_file=gmodel_checkpoint_name) # for debug

        # Needs to be refactored
        bart_tmp = torch.hub.load('pytorch/fairseq', 'bart.large.cnn').to(gmodel_device)
        gmodel.task.build_dataset_for_inference = bart_tmp.task.build_dataset_for_inference
        gmodel.to(gmodel_device)
        print(next(gmodel.parameters()).device)
        #gmodel.to(device=torch.device('cuda:0'))#gmodel_device)
        gmodel.eval()
        #del bart2
    #else: #lambada

    # mapping a class label to a unused token (temporary)
    class_selection_dict = get_class_selection_weights(df_train_sent_group_by_label, ARGS.beta)    
    class_list = [36938, 37842, 38214, 39253, 39446, 39714, 39753, 39756, 39821, 40241, 41297, 42090, 42424, 42586, 43038, 43361, 43453, 44320, 45544, 45545, 47198, 47654, 48069, 48396, 49731, 49781]

    for class_id in class_list:
        bart_tokenizer.class_token_tensors.append(torch.tensor([class_id]).to(gmodel_device))

    if ARGS.gmodel == 'bart': # BART_{SPAN}
        print('\n== Bart ==')
        for target_cls in classes:    
            gen_set = set()
            target_cls_counts = df_train_sent_group_by_label[df_train_sent_group_by_label['label'] == target_cls]['counts'].item()
            diff = max_count - target_cls_counts # min(max_count, 2 * target_cls_counts) - target_cls_counts

            print('Class: %s, Diff min(N1 - Ni): %s' % (str(target_cls), str(diff)))
            start_time = time.time()

            while len(new_sentence_dict[target_cls]) < diff:
                if len(new_sentence_dict[target_cls]) % 100 == 0:            
                    print('%s-th sentence generation in class %s' % (len(new_sentence_dict[target_cls]), target_cls))

                # step1: source selection
                element = select_source_sentence(labeled_train_data, target_cls)                

                # step2: generation (masking for 20%)
                try:
                    generated_str, meta_dict = bart_generation(element, sampling=True)
                except RuntimeError:
                    print('CUDA ERROR!!!! Retry')
                    print(element)
                    torch.cuda.empty_cache()
                    raise NotImplementError()
                    continue
                if generated_str in gen_set:
                    print('overlap')
                    continue
                
                # storing results
                new_sentence_dict[target_cls].append(generated_str)
                new_sentence_meta_dict[target_cls].append(meta_dict)
            
                total_gen += 1 
                bart_gen += 1            
            print(target_cls, diff, 'Done', str(time.time() - start_time))

    elif ARGS.gmodel == 'our':
        print('\n== Our ==')
        words_list, word_importances_list = [], []
        token_importance_filename = './data/%s/train_%s_%s_%s_%s_importance.tsv' % (ARGS.dataset, ARGS.data_setting, ARGS.imbalanced_ratio, cmodel_loss_type, str(ARGS.learning_rate))

        label_list, sent_list, token_list, score_list = read_token_importance(token_importance_filename)
        assert len(label_list) == len(sent_list) and len(token_list) == len(score_list) and len(sent_list) == len(token_list)
        #df_processed_data = pd.DataFrame(list(zip(label_list, sent_list, token_list, score_list)), columns = ['label', 'sentence', 'tokens', 'token_importances'])                

        words_list, word_importances_list = batch_merge_subwords(token_list, score_list, sub_word_merge_type)
        df_processed_data = pd.DataFrame(list(zip(label_list, sent_list, words_list, word_importances_list)), columns = ['label', 'sentence', 'words', 'word_importances'])
        if ARGS.source_selection == 'cluster':#ARGS.use_token_importance_file:
            # Parameters (TBR)
            num_of_clusters = num_of_classes * 20 # TBD
            max_iter = 100
            n_init = 1

            df_processed_data.index.name = 'id'
                
            ### Clustering
            documents = []
            for tokens in token_list:
                documents.append(' '.join(tokens))

            ARGS.text_feature = 'BOW' #['BOW', 'TFIDF']
            if ARGS.text_feature == 'TFIDF':
                vectorizer = TfidfVectorizer()
            else:
                vectorizer = CountVectorizer()

            X = vectorizer.fit_transform(documents)

            model = KMeans(n_clusters=num_of_clusters, init='k-means++', max_iter=max_iter, n_init=n_init)
            model.fit(X)
            
            cluster_cnts = [0] * num_of_clusters
            #Counter(model.labels_)
            for cnt in model.labels_:
                cluster_cnts[cnt] += 1

            df_processed_data['cluster_id'] = model.labels_
            
            ### Pre-processing of Masking               
            df_processed_data['source'] = df_processed_data.apply(lambda row : gmodel_input_ids(row), axis=1)
            df_processed_data['is_word_start'] = df_processed_data.apply(lambda row : get_is_word_start(row), axis=1)
            df_processed_data['word_to_bart_token_aligns'] = df_processed_data.apply(lambda row : get_word_to_bart_token_aligns(row), axis=1)
            #df_processed_data.index.name = 'id'
                
        else: # not supported now
            # 여기서 token_importance를 호출하자

            ### Pre-processing of Masking               
            df_processed_data['source'] = df_processed_data.apply(lambda row : gmodel_input_ids(row), axis=1)
            df_processed_data['is_word_start'] = df_processed_data.apply(lambda row : get_is_word_start(row), axis=1)
            df_processed_data['word_to_bart_token_aligns'] = df_processed_data.apply(lambda row : get_word_to_bart_token_aligns(row), axis=1)
            
        for target_cls in classes:    
            target_cls_counts = df_train_sent_group_by_label[df_train_sent_group_by_label['label'] == target_cls]['counts'].item()
            diff = max_count - target_cls_counts
            print('Class: %s, Diff (N1 - Nk): %s' % (str(target_cls), str(diff)))

            if diff == 0: continue
                        
            total_source_count = df_processed_data.shape[0]
            if ARGS.source_selection == 'cluster':
                cur_cluster = -1 
                cluster_source_sequence = []
                for i in range(num_of_clusters):
                    cluster_source_sequence.append(torch.randperm(cluster_cnts[i]).tolist())            
                assert sum(cluster_cnts) == total_source_count            
            start_time = time.time()
            #source_importance here
            print('Diff', diff)
            prev_k = -1
            k = 0
            is_rejected = False
            
            source_select_count = 0
            source_set = set()
            gen_set = set()
            #for debugging
            num_of_src = 0
            num_of_tgt = 0
            num_of_oth = 0

            while k < diff:
#                 print('%s/%s-th sentence in class %s\nsource_select_count: %d\nTotal causality test: %d\nSRC: %d\nTGT: %d\nOTH: %d\n' \
#                       % (k, diff, target_cls, source_select_count, (num_of_src+num_of_tgt+num_of_oth), num_of_src, num_of_tgt, num_of_oth), end="\r")

                #print('%s-th sentence in class %s (%s)' % (k, target_cls, source_select_count))
                #if source_select_count % 100 == 0:
                #    print('%s-th source select in class %s' % (k, target_cls))


                # step 1-1: source class selection
                #source_cls = select_source_class(class_selection_dict, target_cls)
                
                if ARGS.source_selection == 'cluster':
                    # step1-2 source element selection
                    source_element = None
                    while source_select_count < total_source_count:
                        if is_rejected is False:
                            cur_cluster += 1
                            cur_cluster %= num_of_clusters  
                        source_cls = cur_cluster
                        
                        if len(cluster_source_sequence[cur_cluster]) > 0:                        
                            source_index = cluster_source_sequence[cur_cluster].pop()
                            source_select_count += 1
                            element = df_processed_data[df_processed_data['cluster_id'] == cur_cluster].iloc[source_index]

                            if element['label'] == target_cls: 
                                is_rejected = True
                                continue                            
                            else:
                                source_element = element
                                break
                        else:
                            is_rejected = False

                    if total_source_count == source_select_count:
                        print('%s/%s-th sentence in class %s (BART)' % (k, diff, target_cls), end="\r")
                        # step1: source selection
                        element = select_source_sentence(labeled_train_data, target_cls)

                        # step2: generation (masking for 20%)
                        generated_str, meta_dict = bart_generation(element, sampling=True)
                        
                        # filter overlap
                        if generated_str in gen_set:
                            reject_count += 1
                            continue

                        gen_set.add(generated_str)                
                        meta_dict['source_label'] = meta_dict['label']
                        meta_dict['target_label'] = meta_dict['label']
                        meta_dict['source_sentence'] = meta_dict['sentence']
                        meta_dict['model'] = 'bart'

                        del meta_dict['label']
                        del meta_dict['sentence']
                
                        new_sentence_dict[target_cls].append(generated_str)            
                        new_sentence_meta_dict[target_cls].append(meta_dict)
                        bart_gen += 1
            
                        total_gen += 1 
                        k += 1
                        continue
                    #source_element = select_source_sentence(df_processed_data, cur_cluster, type='cluster')

                    # step2: token importance
                    #tokens = source_element['tokens'].item()
                    #token_importances = source_element['token_importances'].item()[target_cls] # TBR for ATIS
                    #source_element = select_source_sentence(labeled_train_data, source_cls)
                    words = source_element['words']
#                     print(type(source_element['word_importances']))
#                     print(source_element['word_importances'])
                    if False:
                        tokens, token_importances = get_source_token_importances(cmodel, source_element, target_cls)
                        words, word_importances = merge_subwords(tokens, token_importances, sub_word_merge_type) 
                        print(word_importances)
                    else:
                        word_importances = source_element['word_importances'][target_cls] # TBR for ATIS
#                         print(source_element['word_importances'])
#                         print(source_element['word_importances'][target_cls])
#                         print(source_element['word_importances'][source_cls])
#                         print(source_element)
#                         assert 1 == 2
#                     print(word_importances)
#                     print(source_element)
#                     with open('cluster.pkl', 'wb') as fp:
#                         pickle.dump(source_element, fp, pickle.HIGHEST_PROTOCOL)
                else: # original                          
                    # step1: source selection
                    source_cls = select_source_class(class_selection_dict, target_cls)
                    source_element = select_source_sentence(df_processed_data, source_cls).iloc[0]
#                     print(source_element)
#                     with open('random_seed.pkl', 'wb') as fp:
#                         pickle.dump(source_element, fp, pickle.HIGHEST_PROTOCOL)
#                     print(df_processed_data.keys())
#                     print(type(source_element['word_importances']))
#                     print(source_element['word_importances'])
                    #print(source_element)
                    # step2: token importance
                    words = source_element['words']
                    word_importances = list(source_element['word_importances'])[target_cls] # TBR for ATIS
#                     tokens, token_importances = get_source_token_importances(cmodel, source_element, target_cls)
#                     words, word_importances = merge_subwords(tokens, token_importances, sub_word_merge_type) 
#                     print(word_importances)
#                 if not ARGS.use_token_importance_file:
#                     random.shuffle(word_importances)

                # step3: masking, generation, rejection
                #tuples = [(i, tokens[i], token_importances[i]) for i in range(len(tokens))]
                #sorted_tuples = sorted(tuples, key=lambda x: -x[2], reverse=True)
                tuples = [(i, words[i], word_importances[i]) for i in range(len(words))]
                sorted_tuples = sorted(tuples, key=lambda x: -x[2], reverse=True)
                
                #source = add_whole_word_mask(source, ARGS.mask_ratio) # masking                              
                iteration = 0
                rejected_by_mask_threshold = True
                is_rejected = True

                source = source_element['source'].clone()
                source[1] = bart_tokenizer.class_token_tensors[target_cls]
                source_length = len(source)
                is_word_start = source_element['is_word_start'].clone()
                word_to_bart_token_aligns = source_element['word_to_bart_token_aligns']

                to_keep = torch.ones(len(source), dtype=torch.bool)
                prev_indices = torch.tensor([])
                
                n_times_mask = True #for SENT
                for index, word, word_importance in sorted_tuples[:-1]:
                    if n_times_mask and len(sorted_tuples) > 20 and index % int(len(sorted_tuples)/20) != 0:
                        continue
                    indices = torch.tensor(word_to_bart_token_aligns[index])
                    if len(prev_indices) > 0:
                        indices = torch.cat((indices , prev_indices), dim=-1)
                    prev_indices = indices.clone()

                    assert source_length - 1 not in indices    
                    source[indices] = bart_tokenizer.mask_idx

                    while indices.size(0) > 0:
                        next_masked = source[indices + 1] == bart_tokenizer.mask_idx
                        indices_next_masked = indices[next_masked] + 1
                        to_keep[indices_next_masked] = 0

                        uncompleted = is_word_start[indices + 1] == 0
                        indices = indices[uncompleted] + 1
                        to_keep[indices] = 0

                    gmodel_input_ids = source[to_keep]
                    if tuple(gmodel_input_ids) in source_set:
                        reject_count += 1
                        iteration += 1
                        continue
                    
                    source_set.add(tuple(gmodel_input_ids))
                    gmodel_input_ids = gmodel_input_ids.unsqueeze(0).to('cpu')                    
                    
#                     print('gmodel_input_ids', gmodel_input_ids)
                    generated_output = gmodel.generate(gmodel_input_ids, sampling=True, beam=1, sampling_topk=0, sampling_topp=0.9, max_len_b=256, min_len=1, no_repeat_ngram_size=3)
                    generated_ids = generated_output[0]['tokens'].to('cpu')
                    rightmost_zero_index = np.argwhere(generated_ids== 0)[-1][-1] 
                    generated_str = bart_tokenizer.decode(generated_ids[rightmost_zero_index + 2:-1])
                    
                    if generated_str in gen_set:
                        reject_count += 1
                        iteration += 1
                        continue
                    
                    gen_set.add(generated_str)
                                        
                    
                    #step 3-3: rejection    
                    input_ids = torch.tensor(tokenizer.encode(generated_str, add_special_tokens=True))
                    input_mask = torch.ones(input_ids.shape, dtype=torch.long)
                    token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long)

                    output = cmodel.forward(input_ids.unsqueeze(0), input_mask.unsqueeze(0), token_type_ids.unsqueeze(0))
                    pred = torch.argmax(output, dim=1)#.cpu().numpy()
#                     print('source label', source_element['label'])
#                     print('target label', target_cls)
# #                     print(target_cls)
# #                     print(generated_str)
# #                     print(output)
#                     print('prediction', pred)
                    if pred == source_element['label']:
                        num_of_src += 1
                    elif pred == target_cls:
                        num_of_tgt += 1
                    else:
                        num_of_oth += 1
                    print('%s/%s-th sentence in class %s\tsource_select_count: %d\tTotal causality test: %d\tSRC: %d\tTGT: %d\tOTH: %d' % (k, diff, target_cls, source_select_count, (num_of_src+num_of_tgt+num_of_oth), num_of_src, num_of_tgt, num_of_oth), end="\r")
                        
                    #assert 1 == 2

                    if pred == target_cls and len(generated_str.strip()) > 0: # Accept case
                        # Need to define a fuction 'get_meta_dict()''
                        meta_dict = dict()
                        meta_dict['iteration (# of mask)'] = iteration + 1                
                        meta_dict['source_label'] = source_element['label']
                        meta_dict['source_sentence'] = source_element['sentence']
                        meta_dict['target_label'] = target_cls                
                        #meta_dict['new_source_sentence_all_masks'] = new_source_sentence_all_masks                
                        meta_dict['generated_sentence'] = generated_str
                        meta_dict['model'] = 'our'

                        new_sentence_dict[target_cls].append(generated_str)
                        new_sentence_meta_dict[target_cls].append(meta_dict)                    
                        our_gen += 1
                        rejected_by_mask_threshold = False
                        is_rejected = False
                        k += 1
                        print(generated_str.encode('utf-8'))
                        break
                
                    elif len(generated_str.strip()) == 0:
                        continue
                    
                    iteration += 1
                    reject_count += 1
                    
                    #DISCARD
                    if pred != source_element['label'] and pred != target_cls:
                        break
            
                #if rejected_by_mask_threshold == True:
                #    # step1: source selection
                #    element = select_source_sentence(labeled_train_data, target_cls)

                #    # step2: generation (masking for 20%)
                #    generated_str, meta_dict = bart_generation(element, sampling=True)
                
                #    meta_dict['source_label'] = meta_dict['label']
                #    meta_dict['target_label'] = meta_dict['label']
                #    meta_dict['source_sentence'] = meta_dict['sentence']
                #    meta_dict['model'] = 'bart'

                #    del meta_dict['label']
                #    del meta_dict['sentence']
                
                #    new_sentence_dict[target_cls].append(generated_str)            
                #    new_sentence_meta_dict[target_cls].append(meta_dict)
                #    bart_gen += 1
            
                #total_gen += 1 

            print(target_cls, diff, 'Done', str(time.time() - start_time))
    
    elif ARGS.gmodel == 'lambada':  # Need to be refactored
        for batch in loader_dict['train']:
            output = cmodel(batch['input_ids'], batch['input_mask'], batch['token_type_ids'])
            pred = torch.argmax(output, dim=1).cpu().numpy()

            for i in range(len(batch['label'])):
                if batch['label'][i] == pred[i]:
                    print(batch['label'][i].item(), tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                    new_sentence_dict[batch['label'][i].item()].append(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                    total_gen += 1
                else:
                    reject_count += 1

    else:
        NotImplementedError()

    print('Total rejection count:', reject_count)
    print('Total generation count:', total_gen)
    print(' >> Bart generation count:', bart_gen)
    print(' >> Our generation count:', our_gen)

    filename = './data/' + ARGS.dataset + '/aug_%s_%s_%s.csv' % (ARGS.data_setting, ARGS.cmodel, ARGS.gmodel) # cmodel_name
    filename_meta = './data/' + ARGS.dataset + '/aug_%s_%s_%s.meta' % (ARGS.data_setting, ARGS.cmodel, ARGS.gmodel) # cmodel_name
    
    fwrite = open(filename, 'w', encoding='utf-8')
    for cls, sentences in new_sentence_dict.items():
        for sentence in sentences:
            fwrite.write(str(cls) + '\t' + sentence + '\n')
    fwrite.close()

    fwrite = open(filename_meta, 'w', encoding='utf-8')
    for cls, meta_dict_list in new_sentence_meta_dict.items():
        for meta_dict in meta_dict_list:
            fwrite.write(str(meta_dict) + '\n')
    fwrite.close()    
    
