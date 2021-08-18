#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import csv
import os

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils_m2m import random_perturb, make_step, inf_data_gen, Logger
from utils_m2m import soft_cross_entropy, classwise_loss, LDAMLoss, FocalLoss

# GENT
from utils_en import *
from config_our import *
from model import BERTClassifier

N_CLASSES = num_of_classes

def save_result(result_filename, result_info_filename, ARGS, test_acc, test_balanced_acc, test_loss):
    result_id = 0
    if os.path.exists(result_filename):
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

LOGNAME = 'Imbalance_' + LOGFILE_BASE
logger = Logger(LOGNAME)
LOGDIR = logger.logdir

LOG_CSV = os.path.join(LOGDIR, f'log_{SEED}.csv')
LOG_CSV_HEADER = [
    'epoch', 'train loss', 'gen loss', 'train acc', 'gen_acc', 'prob_orig', 'prob_targ',
    'test loss', 'major test acc', 'neutral test acc', 'minor test acc', 'test acc', 'f1 score'
]
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(LOG_CSV_HEADER)


def save_checkpoint(acc, model, optim, epoch, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.t7'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '.t7'

    ckpt_path = os.path.join(LOGDIR, ckpt_name)
    torch.save(state, ckpt_path)


def train_epoch(model, criterion, optimizer, data_loader, logger=None):
    print('Start train_epoch()')
    model.train()
    model.bert.train()

    train_loss = 0
    correct = 0
    total = 0

    # Normalizer X
    # Optimizer (SGD, not ADAMW)

    for i, batch in enumerate(data_loader):  #inputs, targets in tqdm(data_loader):
        #print('batch', i)
        # For SMOTE, get the samples from smote_loader instead of usual loader
        if epoch >= ARGS.warm and ARGS.smote:
            inputs, targets = next(smote_loader_inf)

        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = len(batch['input_ids'])#inputs.size(0)

        #outputs, _ = model(normalizer(inputs)) # !!!!!
        outputs = model.forward(batch['input_ids'], batch['input_mask'], batch['token_type_ids'])
        loss = model.criterion(outputs, batch['label']).sum()

        train_loss += loss.item()
        predicted = outputs.max(1)[1]
        total += batch_size
        correct += sum_t(predicted.eq(batch['label']))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        #break # JW

    msg = 'Loss: %.3f| Acc: %.3f%% (%d/%d)' % \
          (train_loss / total, 100. * correct / total, correct, total)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return train_loss / total, 100. * correct / total


def uniform_loss(outputs):
    weights = torch.ones_like(outputs) / N_CLASSES

    return soft_cross_entropy(outputs, weights, reduction='mean')


def classwise_loss(outputs, targets):
    out_1hot = torch.zeros_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), 1)
    return (outputs * out_1hot).sum(1).mean()


def generation(model_g, model_r, inputs, seed_targets, targets, p_accept,
               gamma, lam, step_size, random_start=True, max_iter=10):
    model_g.eval()
    model_r.eval()
    criterion = nn.CrossEntropyLoss()

    if random_start:
        random_noise = random_perturb(inputs, 'l2', 0.5)
        inputs = torch.clamp(inputs + random_noise, 0, 1)

    for _ in range(max_iter):
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs_g = model_g.classifier(inputs)
        outputs_r = model_r.classifier(inputs)
        #outputs_g, _ = model_g(normalizer(inputs))
        #outputs_r, _ = model_r(normalizer(inputs))

        #print('Input', inputs.shape)
        #print('Len(Target)', len(targets))
        #print('Len(Seed_Target)', len(seed_targets))
        #print('Len(Output_G)', outputs_g.shape)
        #print('Len(Output_R)', outputs_r.shape)
       
        loss = criterion(outputs_g, targets) + lam * classwise_loss(outputs_r, seed_targets)
        grad, = torch.autograd.grad(loss, [inputs])

        #print('make_step', make_step(grad, 'l2', step_size).shape)
        inputs = inputs - make_step(grad, 'l2', step_size)
        #print('new_input', inputs.shape)
        inputs = torch.clamp(inputs, 0, 1)
        #print('new_input2', inputs.shape)

    inputs = inputs.detach()

    #outputs_g, _ = model_g(normalizer(inputs))
    outputs_g = model_g.classifier(inputs)

    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, targets.view(-1, 1), 1)
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]

    correct = (probs_g >= gamma) * torch.bernoulli(p_accept).byte().to(device)
    model_r.train()

    return inputs, correct


def train_net(model_train, model_gen, criterion, optimizer_train, batch_orig, targets_orig, gen_idx, gen_targets):
    model_train.train()
    model_train.bert.eval()

    #batch_size = inputs_orig.size(0)
    input_ids = batch_orig['input_ids'].clone()
    token_type_ids = batch_orig['token_type_ids'].clone()
    input_mask = batch_orig['input_mask'].clone()
    targets = batch_orig['label'].clone()
    batch_size = len(targets)
    
    _, inputs_orig = model_train.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
    inputs = inputs_orig.clone()

    #inputs = inputs_orig.clone()
    #targets = targets_orig.clone()

    ########################

#     print('gen_idx', gen_idx)
#     print('gen_targets', gen_targets)
    bs = num_of_class_samples_T[targets_orig].repeat(gen_idx.size(0), 1)
    gs = num_of_class_samples_T[gen_targets].view(-1, 1)

#     print('bs, gs', bs, gs)

    delta = F.relu(bs - gs)
    p_accept = 1 - ARGS.beta ** delta
    mask_valid = (p_accept.sum(1) > 0)
    
#     print('delta', delta)
#     print('p_accept', p_accept)

    gen_idx = gen_idx[mask_valid]
    gen_targets = gen_targets[mask_valid]
    p_accept = p_accept[mask_valid]

    #print('N_SAMPLES_PER_CLASS_T', N_SAMPLES_PER_CLASS_T)
    #print('p_accept', p_accept)

    select_idx = torch.multinomial(p_accept, 1, replacement=True).view(-1)
#     print('p_accept', p_accept)
#     print('select_idx', select_idx)
#     print('select_idx.view(-1,1)', select_idx.view(-1,1))
#     print('p_accept.gather(1, select_idx.view(-1, 1)).view(-1)', p_accept.gather(1, select_idx.view(-1, 1)).view(-1))
    p_accept = p_accept.gather(1, select_idx.view(-1, 1)).view(-1)

    seed_targets = targets_orig[select_idx]
    seed_images = inputs_orig[select_idx]

    #print('seed_targets', seed_targets)

    gen_inputs, correct_mask = generation(model_gen, model_train, seed_images, seed_targets, gen_targets, p_accept,
                                          ARGS.gamma, ARGS.lam, ARGS.step_size, True, ARGS.attack_iter)

    ########################

    # Only change the correctly generated samples
    num_gen = sum_t(correct_mask)
    num_others = batch_size - num_gen

    gen_c_idx = gen_idx[correct_mask]
    others_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    others_mask[gen_c_idx] = 0
    others_idx = others_mask.nonzero().view(-1)

    if num_gen > 0:
        gen_inputs_c = gen_inputs[correct_mask]
        gen_targets_c = gen_targets[correct_mask]

        inputs[gen_c_idx] = gen_inputs_c
        targets[gen_c_idx] = gen_targets_c

    outputs = model_train.classifier(inputs) #(normalizer(inputs))
    loss = criterion(outputs, targets)

    optimizer_train.zero_grad()
    loss.mean().backward()
    optimizer_train.step()

    # For logging the training

    oth_loss_total = sum_t(loss[others_idx])
    gen_loss_total = sum_t(loss[gen_c_idx])

    _, predicted = torch.max(outputs[others_idx].data, 1)
    num_correct_oth = sum_t(predicted.eq(targets[others_idx]))

    num_correct_gen, p_g_orig, p_g_targ = 0, 0, 0
    success = torch.zeros(N_CLASSES, 2)

    if num_gen > 0:
        _, predicted_gen = torch.max(outputs[gen_c_idx].data, 1)
        num_correct_gen = sum_t(predicted_gen.eq(targets[gen_c_idx]))
        probs = torch.softmax(outputs[gen_c_idx], 1).data

        p_g_orig = probs.gather(1, seed_targets[correct_mask].view(-1, 1))
        p_g_orig = sum_t(p_g_orig)

        p_g_targ = probs.gather(1, gen_targets_c.view(-1, 1))
        p_g_targ = sum_t(p_g_targ)

    for i in range(N_CLASSES):
        if num_gen > 0:
            success[i, 0] = sum_t(gen_targets_c == i)
        success[i, 1] = sum_t(gen_targets == i)

    return oth_loss_total, gen_loss_total, num_others, num_correct_oth, num_gen, num_correct_gen, p_g_orig, p_g_targ, success


def train_gen_epoch(net_t, net_g, criterion, optimizer, data_loader):
    net_t.train()
    net_g.eval()

    oth_loss, gen_loss = 0, 0
    correct_oth = 0
    correct_gen = 0
    total_oth, total_gen = 1e-6, 1e-6
    p_g_orig, p_g_targ = 0, 0
    t_success = torch.zeros(N_CLASSES, 2)

    for i, batch in enumerate(data_loader):#inputs, targets in tqdm(data_loader):
        batch_size = len(batch['input_ids']) #inputs.size(0)
        #inputs, targets = #inputs.to(device), targets.to(device)
        targets = batch['label']
        #print('targets', targets)

        # Set a generation target for current batch with re-sampling
        if ARGS.imb_type != 'none':  # Imbalanced
            # Keep the sample with this probability
#             print('N_SAMPLES_PER_CLASS_T', N_SAMPLES_PER_CLASS_T)
#             print('N_SAMPLES_PER_CLASS', N_SAMPLES_PER_CLASS)
#             print('num_of_class_samples', num_of_class_samples)
#             print(targets)
            gen_probs = num_of_class_samples_T[targets] / torch.max(num_of_class_samples_T)
#             print('gen_probs', gen_probs)
            gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()    # Generation index
#             print('gen_index', gen_index)
            gen_index = gen_index.view(-1)
#             print('gen_index2', gen_index)
            gen_targets = targets[gen_index]
#             print('gen_targets', gen_targets)
        else:   # Balanced
            gen_index = torch.arange(batch_size).view(-1)
            gen_targets = torch.randint(N_CLASSES, (batch_size,)).to(device).long()

        #t_loss, g_loss, num_others, num_correct, num_gen, num_gen_correct, p_g_orig_batch, p_g_targ_batch, success \
        #    = train_net(net_t, net_g, criterion, optimizer, inputs, targets, gen_index, gen_targets)
        t_loss, g_loss, num_others, num_correct, num_gen, num_gen_correct, p_g_orig_batch, p_g_targ_batch, success \
            = train_net(net_t, net_g, criterion, optimizer, batch, targets, gen_index, gen_targets)

        oth_loss += t_loss
        gen_loss += g_loss
        total_oth += num_others
        correct_oth += num_correct
        total_gen += num_gen
        correct_gen += num_gen_correct
        p_g_orig += p_g_orig_batch
        p_g_targ += p_g_targ_batch
        t_success += success

    res = {
        'train_loss': oth_loss / total_oth,
        'gen_loss': gen_loss / total_gen,
        'train_acc': 100. * correct_oth / total_oth,
        'gen_acc': 100. * correct_gen / total_gen,
        'p_g_orig': p_g_orig / total_gen,
        'p_g_targ': p_g_targ / total_gen,
        't_success': t_success
    }

    msg = 't_Loss: %.3f | g_Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_gen: %.3f%% (%d/%d) ' \
          '| Prob_orig: %.3f | Prob_targ: %.3f' % (
        res['train_loss'], res['gen_loss'],
        res['train_acc'], correct_oth, total_oth,
        res['gen_acc'], correct_gen, total_gen,
        res['p_g_orig'], res['p_g_targ']
    )
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return res


if __name__ == '__main__':
    TEST_ACC = 0  # best test accuracy
    BEST_VAL = 0  # best validation accuracy
    test_acc = 0
    test_balanced_acc = 0
    test_loss = 0

    # Weights for virtual samples are generated
    logger.log('==> Building model: %s' % MODEL)

    #net = BERTClassifier(bert, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name=cmodel_loss_type, num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=model_name)
    #net.name = model_name
    #model_load_result = net.load()
    #assert model_load_result == True

    #model_name = '%s_%s_%s_classifier_%s_%s_%s_%s.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), ARGS.loss_type, ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel)    

    model_name = '%s_%s_%s_classifier_%s_%s_%s_%s_M2M.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), ARGS.loss_type, ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel)
    net = BERTClassifier(bert, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name=ARGS.loss_type, num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=model_name, random_seed=ARGS.random_seed)
    net.name = 'M2M_model'

    #net = models.__dict__[MODEL](N_CLASSES)
    #net_seed = models.__dict__[MODEL](N_CLASSES)

    net = net.to(device)
    #optimizer = optim.SGD(net.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=ARGS.decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}            
        ]        
    optimizer = AdamW(optimizer_grouped_parameters, lr=net.learning_rate)
    t_total = df_train_data_filtered.shape[0] * net.batch_size * net.num_of_epoch
    warmup_step = int(t_total * net.warmup_ratio)        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    #if ARGS.net_g is None: raise NotImplementedError()        
    ckpt_g = '%s_%s_%s_classifier_%s_%s_%s_%s.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), 'CE', ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel)
    bert2, tokenizer2 = get_bert()
    del tokenizer2
    #print(ckpt_g)
    net_seed = BERTClassifier(bert2, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name='CE', num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=ckpt_g, random_seed=ARGS.random_seed)
    # Load checkpoint.
    if ARGS.resume:
        logger.log('==> Resuming from checkpoint..')
        net_seed.load()
    net_seed.to(device)
#     else:
#         if ARGS.net_g is None: raise NotImplementedError()

    '''
    if N_GPUS > 1:
        logger.log('Multi-GPU mode: using %d GPUs for training.' % N_GPUS)
        net = nn.DataParallel(net)
        net_seed = nn.DataParallel(net_seed)
    elif N_GPUS == 1:
        logger.log('Single-GPU mode.')
    '''
    if ARGS.warm < START_EPOCH and ARGS.over:
        raise ValueError("warm < START_EPOCH")

    SUCCESS = torch.zeros(EPOCH, N_CLASSES, 2)
    test_stats = {}




    for epoch in range(START_EPOCH, EPOCH):
        logger.log(' * Epoch %d: %s' % (epoch, LOGDIR))

        #adjust_learning_rate(optimizer, LR, epoch)

        # JW consider
        if epoch == ARGS.warm and ARGS.over:
            if ARGS.smote:
                logger.log("=============== Applying smote sampling ===============")
                smote_loader, _, _ = get_smote(DATASET, N_SAMPLES_PER_CLASS, BATCH_SIZE, transform_train, transform_test)
                smote_loader_inf = inf_data_gen(smote_loader)
            else:
                logger.log("=============== Applying over sampling ===============")
                train_loader, _, _ = get_oversampled(DATASET, N_SAMPLES_PER_CLASS, BATCH_SIZE,
                                                     transform_train, transform_test)
        #if epoch == ARGS.warm and ARGS.over:
        #    train_loader = 

        ## For Cost-Sensitive Learning ##

        if ARGS.cost and epoch >= ARGS.warm:
            beta = ARGS.eff_beta
            if beta < 1:
                effective_num = 1.0 - np.power(beta, num_of_class_samples)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
            else:
                per_cls_weights = 1 / np.array(num_of_class_samples)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_of_class_samples)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        else:
            per_cls_weights = torch.ones(len(num_of_class_samples)).to(device)

        ## Choos a loss function ##

        if ARGS.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').to(device)
        elif ARGS.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=ARGS.focal_gamma, reduction='none').to(device)
        elif ARGS.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=num_of_class_samples, max_m=0.5, s=30, weight=per_cls_weights,
                                 reduction='none').to(device)
        else:
            raise ValueError("Wrong Loss Type")

        ## Training ( ARGS.warm is used for deferred re-balancing ) ##

        if epoch >= ARGS.warm and ARGS.gen:
        #if False:
            train_stats = train_gen_epoch(net, net_seed, criterion, optimizer, train_loader)
            SUCCESS[epoch, :, :] = train_stats['t_success'].float()
            logger.log(SUCCESS[epoch, -10:, :])
            np.save(LOGDIR + '/success.npy', SUCCESS.cpu().numpy())
        else:
            train_loss, train_acc = train_epoch(net, criterion, optimizer, train_loader, logger)
            train_stats = {'train_loss': train_loss, 'train_acc': train_acc}
            if epoch == 159:
                save_checkpoint(train_acc, net, optimizer, epoch, True)

        ## Evaluation ##

        val_eval = evaluate(net, val_loader, logger=logger, _N_CLASSES = num_of_classes)
        val_acc = val_eval['acc']
        val_acc_class = val_eval['class_acc'].mean()
        
        if ARGS.dataset == 'ATIS':
            tmp_cls = [0, 1, 3, 6, 8, 9, 11, 13, 14, 15, 17, 19, 20, 21, 23, 24, 25]
            val_acc_class = val_eval['class_acc'][tmp_cls].mean()
            print(val_eval['class_acc'][tmp_cls])
        print(val_acc, val_acc_class)
        if val_acc_class >= BEST_VAL:
            BEST_VAL = val_acc_class
            test_stats = evaluate(net, test_loader, logger=logger, _N_CLASSES = num_of_classes)
            TEST_ACC = test_stats['acc']
            TEST_ACC_CLASS = test_stats['class_acc']
            test_balanced_acc = test_stats['class_acc'].mean().item()
            test_acc = test_stats['acc']
            test_loss = test_stats['loss']

            if ARGS.dataset == 'ATIS':
                tmp_cls = [0, 1, 3, 6, 8, 9, 11, 13, 14, 15, 17, 19, 20, 21, 23, 24]
                TEST_ACC_CLASS = test_stats['class_acc'][tmp_cls]
                print(TEST_ACC_CLASS)
                test_balanced_acc = test_stats['class_acc'][tmp_cls].mean().item()                
                save_checkpoint(TEST_ACC, net, optimizer, epoch)
                logger.log("========== Class-wise test performance ( avg : {} ) ==========".format(test_balanced_acc))
                np.save(LOGDIR + '/classwise_acc.npy', TEST_ACC_CLASS.cpu())
            else:
                save_checkpoint(TEST_ACC, net, optimizer, epoch)
                logger.log("========== Class-wise test performance ( avg : {} ) ==========".format(test_balanced_acc))
                np.save(LOGDIR + '/classwise_acc.npy', TEST_ACC_CLASS.cpu())

                

        def _convert_scala(x):
            if hasattr(x, 'item'):
                x = x.item()
            return x

        log_tr = ['train_loss', 'gen_loss', 'train_acc', 'gen_acc', 'p_g_orig', 'p_g_targ']
        log_te = ['loss', 'major_acc', 'neutral_acc', 'minor_acc', 'acc', 'f1_score']

        log_vector = [epoch] + [train_stats.get(k, 0) for k in log_tr] + [test_stats.get(k, 0) for k in log_te]
        log_vector = list(map(_convert_scala, log_vector))

        with open(LOG_CSV, 'a') as f:
            logwriter = csv.writer(f, delimiter=',')
            logwriter.writerow(log_vector)

    logger.log(' * %s' % LOGDIR)
    logger.log("Best Accuracy : {}".format(TEST_ACC))
    
    
    result_filename = '%s/%s_%s_%s_%s_%s_%s.res' % (ARGS.result_path, ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.loss_type, ARGS.gmodel, 'M2m')
    result_info_filename = '%s/%s_%s_%s_%s_%s_%s.info' % (ARGS.result_path, ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.loss_type, ARGS.gmodel, 'M2m')
    
    print('saved in', result_filename )
    save_result(result_filename, result_info_filename, ARGS, test_acc, test_balanced_acc, test_loss)   


