import os
import glob
import lmdb
import numpy as np

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def create_dataset(inputPath, outputPath):
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}

    with env.begin(write=False) as txn :
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None: 
            cnt = 1
        else:
            cnt = int(num_samples)    

    for image_path in glob.iglob(os.path.join(inputPath, "**"), recursive=True):
        print(f'{image_path}')
        if os.path.isfile(image_path): 
            _ , file_name = os.path.split(image_path)
            label = ''
        else:
            continue

        with open(image_path, 'rb') as f:
            image_bin = f.read()

        image_key = 'image-%09d'.encode() % cnt
        label_key = 'label-%09d'.encode() % cnt
        cache[image_key] = image_bin
        cache[label_key] = label.encode()

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d' % (cnt))
        cnt += 1

    cache['num-samples'.encode()] = str(cnt-1).encode()
    write_cache(env, cache)       
    print('Created dataset with %d samples' % cnt)


if __name__ == '__main__':
    source_path = '../../data/nullee_invoice/TR_data_patch/'    
    destination_path = '../../data/nullee_invoice/TR_data_lmdb'
    for folder_name in os.listdir(source_path):
        data_path = os.path.join(source_path, folder_name)
        target_path = os.path.join(destination_path, folder_name)
        create_dataset(data_path, target_path)    
