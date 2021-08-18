import torch
import torch.nn as nn
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer
import model_utils
from model import BERTClassifier, get_linear_schedule_with_warmup
from utils_en import *
from config import *

def get_bert():
    model = BertModel4Mix.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    
    return model, tokenizer


class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids,  input_ids2=None, l=None, mix_layer=1000, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        
        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:

            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)

            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

                       
        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1-l)*hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


    
    
    
class MixText(BERTClassifier):
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
        super().__init__(bert,
                 hidden_size,
                 dr_rate,
                 params,
                 batch_size,
                 warmup_ratio,
                 num_of_epoch,
                 max_grad_norm,
                 learning_rate,
                 criterion_name,                 
                 num_of_classes,
                 num_of_class_samples, # only for LDAM
                 device,
                 model_name[:-5]+'_TMix.ckpt',
                 random_seed, 
                 cancel_pos,
                 additive)
        
    def tmix_criterion(self, outputs_x, targets_x):
        return - \
            torch.mean(torch.sum(F.log_softmax(
                outputs_x, dim=1) * targets_x, dim=1))
        
    def forward(self, input_ids, input_mask, token_type_ids, input_ids_2=None, input_mask_2=None, token_type_ids_2=None, l=None, mix_layer=1000):
        x = input_ids#(input_ids, input_mask, token_type_ids)
        x2 = input_ids_2#(input_ids_2, input_mask_2, token_type_ids_2)

        if x2 is not None:
            _, pooler = self.bert(x, x2, l, mix_layer)

        else:
            _, pooler = self.bert(x)

        if self.dr_rate:
            out = self.dropout(pooler)        
        
        return self.classifier(out)
    
    
    def train_tmix(self, batch):        
        def get_idxs(batch_labels, labels):
            idxs = None
            for t in labels:
                tmp_idxs = (batch_labels == t).nonzero(as_tuple=True)[0].view(-1)
                if idxs is None:
                    idxs = tmp_idxs
                else:
                    idxs = torch.cat([idxs, tmp_idxs])
            return idxs
        
        
        head_idxs = get_idxs(batch['label'], ARGS.HEAD_labels)
        tail_idxs = get_idxs(batch['label'], ARGS.TAIL_labels)
        
        min_len = min(len(head_idxs), len(tail_idxs))
        tail_idxs = tail_idxs[:min_len]
        head_idxs = head_idxs[:min_len]
        assert len(tail_idxs) == len(head_idxs)
        
        def one_hot_embedding(labels, num_classes):
            """Embedding labels to one-hot form.

            Args:
              labels: (LongTensor) class labels, sized [N,].
              num_classes: (int) number of classes.

            Returns:
              (tensor) encoded labels, sized [N, #classes].
            """
            y = torch.eye(num_classes) 
            return y[labels]
                
        input_a = torch.index_select(batch['input_ids'], dim=0, index=head_idxs)
        target_a = torch.index_select(batch['label'], dim=0, index=head_idxs)
        target_a = one_hot_embedding(target_a, \
                                     num_classes=ARGS.TOTAL_labels)
        
        input_b = torch.index_select(batch['input_ids'], dim=0, index=tail_idxs)
        target_b = torch.index_select(batch['label'], dim=0, index=tail_idxs)
        target_b = one_hot_embedding(target_b, \
                                     num_classes=ARGS.TOTAL_labels)
        
        
        alpha = 16
        l = np.random.beta(alpha, alpha)
        
        mix_layers_set = [7, 9, 12]
        mix_layer = np.random.choice(mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1
        
        
        output = self.forward(input_a, None, None, 
                              input_b, None, None, 
                              l=l, mix_layer=mix_layer)
                
        mixed_target = l * target_a + (1 - l) * target_b
                
        return self.tmix_criterion(output, mixed_target)
        
    
    def train_step(self, batch, mix=0):
        if mix == 0:
            output = self.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids'])
            loss = self.criterion(output, batch['label']).sum()            
        else:
            loss = self.train_tmix(batch)
            
        return loss
        
    def do_valid_test(self, valid_data, test_data, tmp_best_score, always_eval_test, epoch):
        self.eval()

        tmp_score, valid_balanced_score, valid_loss, report = self.evaluation(valid_data)            

        save_condition = tmp_best_score is None or tmp_best_score <= valid_balanced_score
        if always_eval_test or save_condition:
            print('=== Validation loss, accuracy, balanced_accuracy at epoch %s: %s, %s, %s [SAVED]' \
                  % (str(epoch), str(round(valid_loss, 4)), round(tmp_score, 4), round(valid_balanced_score, 4)))
            tmp_best_score = valid_balanced_score                
            torch.cuda.empty_cache()
            test_score, test_balanced_score, test_loss, report = self.evaluation(test_data)
            print('==== Test loss, accuracy, balanced_accuracy at epoch %s: %s, %s, %s' \
                  % (str(epoch), str(round(test_loss, 4)), round(test_score, 4), round(test_balanced_score, 4)))
            if save_condition:
                self.score = test_balanced_score
                self.save(self.name)
        else:
            print('=== Validation loss, accuracy, balanced_accuracy at epoch %s: %s, %s, %s [NOT SAVED]' % (str(epoch), str(round(valid_loss, 4)), round(tmp_score, 4), round(valid_balanced_score, 4)))
            
        return tmp_best_score
        
    
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
                if not ARGS.gpu:
                    for key in batch.keys():
                        batch[key] = batch[key].cpu()
                for j in range(2):
                    optimizer.zero_grad()
                    
                    loss = self.train_step(batch, j)                    

                    total_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    if (i*2+j) % 40 == 0:
                        print('== Train loss at epoch %s and step %s: %s' \
                              % (str(_), str(i), str(round(total_loss / ((i + 1) * self.batch_size), 4))))

            tmp_best_score = self.do_valid_test(valid_data, test_data, tmp_best_score, always_eval_test, _)
        self.eval()
        return tmp_best_score