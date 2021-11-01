import os
import glob
import lmdb
import numpy as np


def restore(lmdb_path, inputPath, output_path):
    tagging_file = open(output_path, 'w', encoding='utf-8')
    env = lmdb.open(lmdb_path, map_size=1099511627776)

    with env.begin(write=False) as txn :
        num_samples = txn.get('num-samples'.encode())
        cnt = 1
        for image_path in glob.iglob(os.path.join(inputPath, "**"), recursive=True):
            if not os.path.isfile(image_path): 
                continue

            print(f'{image_path}')            
            label_key = 'label-%09d'.encode() % cnt       
            try: 
                label = txn.get(label_key).decode('utf-8')
            except:
                print(f'{image_path} {cnt} is None\n')
                continue 

            tagging_file.write(f'{image_path}\t{label}\n')
            cnt += 1
    tagging_file.close()


if __name__ == '__main__':
    # source_path = '../../data/nullee_invoice/TR_data_lmdb/전자계산서'    
    # source_path = '../../data/nullee_invoice/TR_data_lmdb/전자세금계산서'
    # source_path = '../../data/nullee_invoice/TR_data_lmdb/종이영수증'
    # source_path = '../../data/nullee_invoice/TR_data_lmdb/포스영수증'
    # source_path = '../../data/nullee_invoice/TR_data_lmdb/세금계산서'
    source_path = '../../data/nullee_invoice/TR_data_lmdb/invoice'


    # patch_path = '../../data/nullee_invoice/TR_data_patch/전자계산서'    
    # patch_path = '../../data/nullee_invoice/TR_data_patch/전자세금계산서'
    # patch_path = '../../data/nullee_invoice/TR_data_patch/종이영수증'
    # patch_path = '../../data/nullee_invoice/TR_data_patch/포스영수증'
    # patch_path = '../../data/nullee_invoice/TR_data_patch/세금계산서'
    patch_path = '../../data/nullee_invoice/TR_data_patch/invoice'

    # destination_path = '../../data/nullee_invoice/TD_data_tagging/전사세금계산서.txt'
    destination_path = '../../data/nullee_invoice/TD_data_tagging/invoice.txt'
    # for folder_name in os.listdir(source_path):
    #     data_path = os.path.join(source_path, folder_name)
        # target_path = os.path.join(destination_path, folder_name)
    restore(source_path, patch_path, destination_path)    
