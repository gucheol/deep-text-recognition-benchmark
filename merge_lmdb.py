import os 
import io
import random
from lmdb_helper import MyLMDB, load_class_dictionary, is_valid_label, image_bin_to_pil

def merge_db(root_path, src_db_path, target_path):

    target_db = MyLMDB(target_path, mode='w', map_size=60e9)

    for db_name in src_db_path:
        db_path = os.path.join(root_path, db_name)
        src_db = MyLMDB(db_path, mode='r')
        num_of_samples = src_db.num_of_samples
        for i in range(1, num_of_samples):
            im, label = src_db.read_image_label(i)
            
            if len(label) > 39:
                print(f'skip {label}')
                continue 
            target_db.write_im_label(im, label)
    target_db.close()       


if __name__== '__main__':
    # target = 'merged_handwritten'
    # root = 'data/train/'
    # src = ['handwritten_aihub', 'handwritten_trdg']
    # merge_db(root, src, target)

    # target = 'merged_printed'
    # root = 'data/train/'
    # src = ['printed_aihub', 'printed_trdg']
    # merge_db(root, src, target)

    # target = 'merged_val'
    # root = 'data/val/'
    # src = ['printed_aihub', 'printed_trdg', 'printed_aihub_aug', 'handwritten_aihub', 'handwritten_trdg']
    # merge_db(root, src, target)

    # target = 'merged_printed_aug'
    # root = 'data-backup/train/'
    # src = ['printed_aihub_aug']
    # merge_db(root, src, target)

    # target = 'nullee_train_strip'
    # garbage_path = 'garbage_val'
    # src = 'data/train/nullee_train'


    # target = 'nullee_train_vertical_strip'
    # garbage_path = 'garbage_vertical_val'
    # src = './data/train/nullee_train_vertical'

    # target = 'nullee_synth'
    # garbage_path = 'garbage_nullee_synth'
    # src = '/media/data/nullee_data/gen_synth_lmdb'
    # # src = '/media/data/nullee_data/unity_lmdb/'

    target = 'nullee_val_vertical'
    garbage_path = 'garbage_nullee_vertical'
    src = 'backup/nullee_val_vertical_no_rotate'    

    class_dict = load_class_dictionary('data/train/kr_labels.txt', add_space=True)

    target_db = MyLMDB(target, mode='w', map_size=6e9)
    src_db = MyLMDB(src, mode='r', map_size=6e9)
    strip_db = MyLMDB(garbage_path, mode='w', map_size=6e9)

    num_of_samples = src_db.num_of_samples
    for i in range(1, num_of_samples):
        # if i > 38000:
        #     break

        im, label = src_db.read_image_label(i)
        if len(label) > 39 or not is_valid_label(label.upper(), class_dict) or label.find('#')>=0 or label == '__#DELETED_LABEL#__)':
            strip_db.write_im_label(im, label)
            print(f'skip {label}')
            continue 

        if label == '주식회사 토라스':
            label = '주식회사 토리스'

        pil_im = image_bin_to_pil(im)

        w, h = pil_im.size
        if w*1.5 < h and len(label) > 1 :
            pil_im = pil_im.rotate(90, expand=True)
            with io.BytesIO() as output:
                pil_im.save(output, format="JPEG")
                im = output.getvalue()
        print (f"{label}")
        target_db.write_im_label(im, label)
    target_db.close()
    strip_db.close()
    
import os
import lmdb
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache, load_classes_dictionary


def get_gt_from_file_name(file_name, classes):
    label = ''
    try:
        name = '.'.join(file_name.split('.')[:-1])
        name = name.split('_L_')[1]
        for ch in name:
            if classes.get(ch) is None:
                print('unknown class: ' + ch)
                label += '<UNK>'
            else:
                label += ch
    except IndexError:
        print(f'{file_name} has index error')
        raise IndexError
    return label


def is_valid_label(label, classes):
    for ch in label:
        if classes.get(ch) is None:
            return False
    return True


def copy_lmdb(src_lmdb_env, target_lmdb_env, classes):
    data_cache = {}
    with target_lmdb_env.begin(write=False) as txn:
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None:
            print('target lmdb is empty')
            target_count = 0
        else:
            target_count = int(num_samples)

    with src_lmdb_env.begin(write=False) as txn:
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None:
            print('src lmdb is empty')
            exit()
        src_count = int(num_samples)

    with src_lmdb_env.begin(write=False) as src_txn:
        for idx in range(0, src_count):
            image_key = 'image-%09d'.encode() % idx
            label_key = 'label-%09d'.encode() % idx
            label = src_txn.get(label_key)
            if label is None:
                print(f'{label} has no data')
                continue

            label = label.decode('utf-8')
            label = label.replace(',', '.')
            if not is_valid_label(label, classes):
                print(f'{label} has invalid label ')
                continue

            image_bin = src_txn.get(image_key)
            image_key = 'image-%09d'.encode() % target_count
            label_key = 'label-%09d'.encode() % target_count
            data_cache[label_key] = label.encode()
            data_cache[image_key] = image_bin
            target_count += 1

            if target_count % 1000 == 1:
                writeCache(target_lmdb_env, data_cache)

        data_cache['num-samples'.encode()] = str(target_count - 1).encode()
        writeCache(target_lmdb_env, data_cache)
        print(f'target dataset size: {target_count}')


def merge_dataset(src1_lmdb_path, src2_lmdb_path, target_lmdb_path):
    os.makedirs(target_lmdb_path, exist_ok=True)
    src1_lmdb_env = lmdb.open(src1_lmdb_path, map_size=1099511627776)
    src2_lmdb_env = lmdb.open(src2_lmdb_path, map_size=1099511627776)
    target_lmdb_env = lmdb.open(target_lmdb_path, map_size=1099511627776)
    classes = load_classes_dictionary('data_generation/kr_labels.txt')
    copy_lmdb(src1_lmdb_env, target_lmdb_env, classes)
    copy_lmdb(src2_lmdb_env, target_lmdb_env, classes)

def renew_dataset(src_lmdb_path, target_lmdb_path):
    os.makedirs(target_lmdb_path, exist_ok=True)
    src1_lmdb_env = lmdb.open(src_lmdb_path, map_size=1099511627776)
    target_lmdb_env = lmdb.open(target_lmdb_path, map_size=1099511627776)
    classes = load_classes_dictionary('data_generation/kr_labels.txt')
    copy_lmdb(src1_lmdb_env, target_lmdb_env, classes)

if __name__ == '__main__':
    # merge_dataset('data/real_data_revised/500_real_data', 'data/evaluation/march_real_data_good')
    # merge_dataset('data/real_data_revised/2500_real_data', 'data/evaluation/march_real_data_good', 'data/real_data_lmdb_0412')
    # merge_dataset('data/real_data_revised/500_real_data', 'data/real_data_lmdb_0412', 'data/real_data_lmdb_0412_v2')
    #  merge_dataset('data/real_data_revised/1200_real_data', 'data/real_data_lmdb_0412_v2', 'data/real_data_lmdb_0412_v3')
    # merge_dataset('data/evaluation/real_data_lmdb', 'data/real_data_lmdb_0412_v3', 'data/real_data_lmdb_0412_final')
    # merge_dataset('added_month_day_lmdb', 'data/train/unity', 'unity_added_month_day_lmdb')
    #  renew_dataset('data/evaluation/real_month_day_march_lmdb', 'data/real_month_day_march_lmdb_renew')
    # merge_dataset('/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/real_digits_feb_lmdb/',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb/',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_v2')
    #
    # merge_dataset('/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb-jan/',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_v2/',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_v3')
    # merge_dataset('/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/real_month_day_march_lmdb_renew/',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_v3',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_final_0512')
    # renew_dataset('/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_final_0512',
    #               '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/TextRecognition_lmdb_0512_final')

    merge_dataset('/home/embian/Workspace/deep-text-recognition-benchmark/data/real/real_data_lmdb_0412_final',
                  '/home/embian/Workspace/deep-text-recognition-benchmark/data/real/real_data_lmdb_0512_final',
                  '/home/embian/Workspace/deep-text-recognition-benchmark/data/real/real_data_lmdb_0513_final')
