from config import *
from os import path

if __name__ == '__main__':
    # TREC_step_100_classifier_CE_5e-05_True_our_sample.ckpt    
    print('\n== Load teacher classifier: %s ==' % ARGS.teacher)     
    model_name = 'TREC_%s_100_classifier_CE_5e-05_True_our_sample.ckpt' % ARGS.data_setting
    model = BERTClassifier(bert, hidden_size = 768, dr_rate=ARGS.dr_rate, batch_size=ARGS.batch_size, params=None, num_of_classes=num_of_classes, warmup_ratio=ARGS.warmup_ratio, num_of_epoch=ARGS.num_of_epoch, max_grad_norm=ARGS.max_grad_norm, learning_rate=ARGS.learning_rate, criterion_name=ARGS.loss_type, num_of_class_samples=num_of_class_samples, device=ARGS.device, model_name=model_name, random_seed=ARGS.random_seed)
    model.to(ARGS.device)
    model.load()

    print('== Test BERT classifier ==')
    test_acc, test_balanced_acc, test_loss, result = model.evaluation(loader_dict['test'])
    print(result)
    
