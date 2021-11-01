import os
import lmdb
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache, load_classes_dictionary


def get_gt_from_file_name(file_name, classes):
    label = ''
    try:
        name = '.'.join(file_name.split('.')[:-1])
        # for real
        name = name.split('_L_')[1]
        # for trdg
        # name = name.split('_')[0]
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


def create_dataset(input_path, output_path, check_valid=True, use_space=False):
    os.makedirs(output_path, exist_ok=True)
    lmdb_env = lmdb.open(output_path, map_size=1099511627776)
    classes = load_classes_dictionary('data_generation/kr_labels.txt', use_space)
    cache = {}
    sample_count = 0

    for image_path in glob(os.path.join(input_path, "**"), recursive=True):
        if os.path.isfile(image_path):
            file_name = os.path.basename(image_path)
            label = get_gt_from_file_name(file_name, classes)
        else:
            continue

        if label.find('<UNK>') >= 0:
            continue

        if len(label) > 30:
            print(f'skip {label} {len(label)}')
            continue

        with open(image_path, 'rb') as f:
            image_bin = f.read()

        if check_valid:
            try:
                if not checkImageIsValid(image_bin):
                    print('%s is not a valid image' % image_path)
                    continue
            except:
                print(f'{image_path} error occurred')
                with open(output_path + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occurred error\n' % str(image_path))
                continue

        image_key = 'image-%09d'.encode() % sample_count
        label_key = 'label-%09d'.encode() % sample_count

        cache[image_key] = image_bin
        cache[label_key] = label.encode()
        sample_count += 1
        if sample_count % 1000 == 0:
            writeCache(lmdb_env, cache)
            cache = {}
            print(f'Written {sample_count}')

    cache['num-samples'.encode()] = str(sample_count - 1).encode()
    writeCache(lmdb_env, cache)
    print(f'Created dataset {sample_count}')


if __name__ == '__main__':
    # create_dataset('/home/embian/Workspace/IDReader/result/real_data_good_sample', 'data/evaluation/march_real_data_good/')
    # create_dataset('/home/embian/Workspace/result/digits/', '/home/embian/Workspace/result/digit-lmdb')
    # create_dataset('raw_data/month_day/', 'added_month_day_lmdb')
    # create_dataset('/home/embian/Workspace/IDReader/result/monty_day_digits', 'data/evaluation/real_month_day_lmdb/')
    # create_dataset('/home/embian/Workspace/IDReader/result/monty_day_digits', 'data/evaluation/real_month_day_remain_lmdb/')
    # create_dataset('/home/embian/Workspace/IDReader/result/month_day_digits_march',
    #                'data/evaluation/real_month_day_march_lmdb/')
    # create_dataset('/home/embian/Workspace/data/AnnotationCandidate/TextRecognition/', '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition_lmdb')
    create_dataset('/home/embian/Workspace/data/server_collected_patch/Jan-seletected',
                   '/home/embian/Workspace/data/AnnotationCandidate/TextRecognition_lmdb-jan')