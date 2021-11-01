import os
import glob

def load_class_dictaionry(path):
    class_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        items = line.strip().split()
        try:
            class_dict[items[1]] = items[0]
        except:
            print(items)

    return class_dict

classes_dict = load_class_dictaionry('data_generation/kr_labels.txt')

new_ch = {}
path = 'Z:\\Workspace\\data\\nullee_invoice\\TD_data_tagging_wip\\'
for folder_name in os.listdir(path):
    for file_path in glob.glob(os.path.join(path, folder_name) + '/*.txt'):
         with open(file_path, 'r', encoding='utf-8') as f:
             lines = f.readlines()

         for line in lines:
            items = line.strip().split('\t')
            word = items[0].upper().replace(' ','')
            for ch in word:
                if classes_dict.get(ch) is None:
                    print(f'{file_path} has {ch}')
                    if new_ch.get(ch) is None:
                        new_ch[ch] = 0
                    else:
                        new_ch[ch] += 1


res = sorted(new_ch.items(), key=(lambda x: x[1]), reverse=True)
print(res)

