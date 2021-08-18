from config import *
from os import path

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
    
if __name__ == '__main__':

    print (ARGS.teacher)
    assert ARGS.head_only ^ (ARGS.teacher != None)
    
    print('\n== Load teacher classifier: %s ==' % ARGS.teacher)     
    model = BERTClassifier(bert, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name=ARGS.loss_type, num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=ARGS.teacher, random_seed=ARGS.random_seed)
    model.to(ARGS.device)


    if ARGS.train_bert:
        for param in model.bert.parameters():
            param.requires_grad = True

    if ARGS.gpu:
        model.cuda(ARGS.device)


    if ARGS.head_only or ARGS.teacher:
        model_name = model_name.replace('.ckpt', '_HEAD.ckpt')

    if ARGS.teacher:
        model.load()

        student_name =  model_name[:-5] + '_teacher:' + ARGS.teacher
        print('\n== Train student classifier: %s ==' % student_name)    
        model.name = student_name                
        
        model._train(loader_dict['train'], loader_dict['valid'], loader_dict['test'])
         
    elif ARGS.head_only:
        model._train(loader_dict['HEAD_train'], loader_dict['HEAD_valid'], loader_dict['test'])
    else:
        NotImplementError()

    
    print('Done!')

    print('\n== Show the best model results ==')
    model.load()

    print('== Test BERT classifier ==')
    print('valid')
    model.evaluation(loader_dict['valid'])
    print('HEAD_test')
    model.evaluation(loader_dict['HEAD_test'])
    print('TAIL_test')
    model.evaluation(loader_dict['TAIL_test'])
    print('test')
    test_acc, test_balanced_acc, test_loss, result = model.evaluation(loader_dict['test'])
    print(result)


    #model_name = '%s_%s_%s_classifier_%s_%s_%s_%s_mv%s.ckpt' % (ARGS.dataset, ARGS.data_setting, str(ARGS.imbalanced_ratio), ARGS.loss_type, ARGS.learning_rate, ARGS.data_augment, ARGS.gmodel, str(ARGS.min_valid_data_num))
    # adding mkdir for log directory
    if ARGS.min_valid_data_num != 5:
        result_filename = '%s/%s_%s_%s_%s_%s_minv%s.res' % (ARGS.result_path, ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.loss_type, ARGS.gmodel, str(ARGS.min_valid_data_num))
        result_info_filename = '%s/%s_%s_%s_%s_%s_minv%s.info' % (ARGS.result_path, ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.loss_type, ARGS.gmodel, str(ARGS.min_valid_data_num))
    else:
        result_filename = '%s/%s_%s_%s_%s_%s.res' % (ARGS.result_path, ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.loss_type, ARGS.gmodel)
        result_info_filename = '%s/%s_%s_%s_%s_%s.info' % (ARGS.result_path, ARGS.dataset, ARGS.data_setting, ARGS.cmodel, ARGS.loss_type, ARGS.gmodel)

    if ARGS.head_only:
        result_filename = result_filename.replace('.res', '_HEAD.res')
        result_info_filename = result_info_filename.replace('.info', '_HEAD.info')
 
    if ARGS.teacher:
        result_filename = result_filename.replace('.res', '_student.res')
        result_info_filename = result_info_filename.replace('.info', '_student.info')

    print('saved in', result_filename )
    save_result(result_filename, result_info_filename, ARGS, test_acc, test_balanced_acc, test_loss)
    
