import ast
import json
from collections import defaultdict

if __name__ == '__main__':   
    dataset = 'SNIPS'
    data_setting = 'longtail'
    input_filename = './data/%s/aug_%s_our_our.meta' % (dataset, data_setting)

    fread = open(input_filename, 'r', encoding='utf-8') #, encoding='utf-8')

    class_total_dict = defaultdict(int)
    class_fail_dict = defaultdict(int)
    class_success_dict = defaultdict(int)
    
    mask_ratios = []
    for line in fread.readlines():
        json_data = ast.literal_eval(line.strip())
        target_cls = json_data['target_label']
        
        if json_data['iteration (# of mask)'] == -1:
            class_total_dict[target_cls] += 1
            class_fail_dict[target_cls] += 1
            mask_ratios.append(1.0)
        else:
            mask_ratios.append((json_data['iteration (# of mask)']  )/ (len(json_data['source_sentence'].strip().split()) + 1))
            class_total_dict[target_cls] += 1
    fread.close()

    print( class_total_dict )
    print( class_fail_dict )
    print( sum(mask_ratios) / len(mask_ratios) )
    #print(mask_ratios)
