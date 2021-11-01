import os
import glob
import lmdb
import numpy as np

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def restore_dataset(original_path, current_path):
    original_env = lmdb.open(original_path, map_size=1099511627776)
    current_env = lmdb.open(current_path, map_size=1099511627776)
    cache = {}

    with original_env.begin(write=False) as original_txn :
        num_samples = int(original_txn.get('num-samples'.encode()))
        with current_env.begin(write=True) as current_txn :        
            for cnt in range(1, num_samples + 1):
                label_key = 'label-%09d'.encode() % cnt       
                current_label = current_txn.get(label_key).decode('utf-8')
                original_label = original_txn.get(label_key).decode('utf-8')
                if current_label.find('DELETED') > 0 and original_label != current_label :
                    current_txn.put(label_key, original_label.encode())
                    print(current_label)

if __name__ == '__main__':
    original_path = '../../data/nullee_invoice/세금계산서_원본/세금계산서'    
    current_path = '../../data/nullee_invoice/TR_data_lmdb/세금계산서'    
    restore_dataset(original_path, current_path)    
