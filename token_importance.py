from config import *
from os import path

if __name__ == '__main__':
    print('\n== Train BERT classifier ==')     

    model_name = '%s_%s_%s_classifier_%s_%s_False_None.ckpt' % (ARGS.dataset, ARGS.data_setting, ARGS.imbalanced_ratio, ARGS.loss_type, str(ARGS.learning_rate))

    print('\n== Load Classifier Model: %s ==' % (model_name))
    model = BERTClassifier(bert, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name=ARGS.loss_type, num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=model_name)
    model.name = model_name
    model.to(ARGS.device)
    model_load_result = model.load()
    assert model_load_result == True
    model.eval()
    
    print('\n== Calculate Token Importance: %s ==' % (model_name))

    target_labels = labeled_train_data.label.unique()
    embedding_type = 'position' # ['token', 'position'] 
       
    fwrite = open('./data/%s/train_%s_%s_%s_%s_importance.tsv' % (ARGS.dataset, ARGS.data_setting, ARGS.imbalanced_ratio, ARGS.loss_type, str(ARGS.learning_rate)), 'w', encoding='utf-8')


    for i, batch in enumerate(loader_dict['train']):
        if i % 100 == 0:            
            print('%s-th batch in %s' % (i, len(loader_dict['train'])))

        for input_ids, input_mask, token_type_ids, label in list(zip(batch['input_ids'], batch['input_mask'], batch['token_type_ids'], batch['label'])):
            input_ids = input_ids.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            label = label.unsqueeze(0)
        
            tokenized_text = []
            attention_weights = []
            importances_strs = ''

            for token_id in input_ids[-1][1:]:
                if tokenizer.sep_token_id == token_id: break
                token = tokenizer.decode(int(token_id))
                tokenized_text.append(token.replace(' ', ''))
        
            for target_label in target_labels:
                target_label = torch.tensor(target_label).unsqueeze(0)
                output = model.forward(input_ids, input_mask, token_type_ids)            
                loss = model.criterion(output, target_label)
                model.zero_grad()
                loss.backward()
                torch.cuda.empty_cache()
        
                importances = torch.tensor([])
                total_importances = 0.0
            
                if embedding_type == 'token': # Not used 
                    for token_id in input_ids[-1][1:]:
                        if tokenizer.sep_token_id == token_id: break                
                        importance = torch.norm(model.bert.embeddings.word_embeddings.weight.grad[int(token_id)], 2)                
                        total_importances += importance
                        importances = torch.cat((importances , importance.unsqueeze(0)), dim=-1)
                elif embedding_type == 'position':
                    for token_index in range(1, len(input_ids[-1])):
                        token_id = input_ids[-1][token_index]
                        if tokenizer.sep_token_id == token_id: break                
                    
                        importance = torch.norm(model.bert.embeddings.position_embeddings.weight.grad[token_index], 2)
                        total_importances += importance
                        importances = torch.cat((importances , importance.unsqueeze(0)), dim=-1)
                else:
                    raise Exception('No Embedding type', embedding_type)
            
                max_val = torch.max(importances)
                if max_val > 0:
                    importances /= torch.max(importances)
                
                importances_str = ' '.join(['{:.3f}'.format(i) for i in importances]) 
                importances_strs += '\t' + importances_str            
            fwrite.write('%s\t%s\t%s\n' % (str(label[0].item()), ' '.join(tokenized_text), importances_strs.strip()))

    fwrite.close()
