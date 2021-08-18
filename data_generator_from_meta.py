from config import *
import ast
import json

def select_source_sentence(df_train, source_cls):
    return df_train[df_train['label'] == source_cls].sample()

if __name__ == '__main__':   
    input_filename = './data/%s/aug_%s_our_our_bart.meta' % (ARGS.dataset, ARGS.data_setting)
    output_filename = './data/%s/aug_%s_our_our_sample.csv' % (ARGS.dataset, ARGS.data_setting)

    fwrite = open(output_filename, 'w', encoding='utf-8')
    fread = open(input_filename, 'r', encoding='utf-8') #, encoding='utf-8')

    meta_info_list = []
    
    for line in fread.readlines():
        json_data = ast.literal_eval(line.strip())
        target_cls = json_data['target_label']
        
        if json_data['iteration (# of mask)'] == -1:
            row = select_source_sentence(labeled_train_data, target_cls)
            sentence = row['sentence'].item()
        else:
            sentence = json_data['generated_sentence']
            
        fwrite.write(str(target_cls) + '\t' + sentence +'\n')
    
    fread.close()
    fwrite.close()
    