from config import *
from collections import defaultdict

# I will merge this file to the translation.py 

def generate_M2M_data_lambada(filename_train, filename_test, y_class, df_sent):
    # parameters (need to be replaced by a config file)    
    fwrite_train = open(filename_train, 'w', encoding='utf-8')
    fwrite_test = open(filename_test, 'w', encoding='utf-8')

    # split training & test sentences for each label
    label_sent_train_dict = defaultdict(list)
    for i in range(len(y_class)):
        label_sent = df_sent.loc[df_sent['label'] == i]['sentence'].to_list()        
        label_sent_train_dict[i] = label_sent

    # generate training & test data for each label
    for i in range(len(y_class)):
        if len(label_sent_train_dict[i]) == 0: continue

            
        # training data
        for x in label_sent_train_dict[i]:
            fwrite_train.write(str(i) + '\t' + x + '\n')
            #fwrite_train.write(x + '\n')
        
        for k in range(1):                
            fwrite_test.write(str(i) + '\n')

    fwrite_train.close()
    fwrite_test.close()

def data_conversion(input_file, class_list, class_dict):
    fread = open(input_file, 'r', encoding='utf-8')
    fwrite_sentence = open(input_file + '_sentence', 'w', encoding='utf-8')
    fwrite_label = open(input_file + '_label', 'w', encoding='utf-8')
    fwrite_sentence_encode = open(input_file + '_sentence_encode', 'w', encoding='utf-8')

    for line in fread.readlines():
        label, sentence = line.strip().split('\t')    
        fwrite_sentence.write('%s\n' % (sentence))
        fwrite_label.write('%s\n' % (class_list[int(label)]))    
        new_line = '%s]%s' % (class_dict[class_list[int(label)]], sentence)
        encoded = bart.encode(sentence)[1:-1].tolist()
        if ARGS.dataset == 'SENT':
            encoded = bart.encode(sentence[1:-1]).tolist()            
        else:
            assert(bart.encode(sentence)[1:].tolist() == bart.encode(new_line)[3:].tolist())
        fwrite_sentence_encode.write(str(encoded) + '\n')
        #fwrite_sentence.write(new_line + '\n')

    fread.close()
    fwrite_sentence.close()
    fwrite_label.close()
    fwrite_sentence_encode.close()

if __name__ == '__main__':   

    print('\n== Load gmodel ==')    
    bart_model_path = './data/model/bart.large/'
    tokenizer = BartTokenizer
    bart = BARTModel.from_pretrained(bart_model_path, checkpoint_file='model.pt')
    bart.eval()

    print('\n== Train Data Generation for gmodel ==')     

    # mapping a class label to a unused token (temporary)
    class_list = [36938, 37842, 38214, 39253, 39446, 39714, 39753, 39756, 39821, 40241, 41297, 42090, 42424, 42586, 43038, 43361, 43453, 44320, 45544, 45545, 47198, 47654, 48069, 48396, 49731, 49781]
    class_dict = dict()

    for cl in class_list:
        class_dict[cl] = bart.decode(torch.tensor([cl]))

    filename_train = './data/%s/m2m_train_%s' % (ARGS.dataset, ARGS.data_setting)
    filename_valid = './data/%s/m2m_valid_%s'% (ARGS.dataset, ARGS.data_setting)
    filename_test = './data/%s/m2m_test_%s' % (ARGS.dataset, ARGS.data_setting)
    
    generate_M2M_data_lambada(filename_train, filename_test, y_class, labeled_train_data)
    generate_M2M_data_lambada(filename_valid, filename_test, y_class, labeled_valid_data)

    data_conversion('./data/%s/m2m_train_%s' % (ARGS.dataset, ARGS.data_setting), class_list, class_dict)
    data_conversion('./data/%s/m2m_valid_%s'% (ARGS.dataset, ARGS.data_setting), class_list, class_dict)
    
    print('Done!')
    


    

