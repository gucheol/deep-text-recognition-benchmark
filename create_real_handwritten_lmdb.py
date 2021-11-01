import os
import json
import random
from lmdb_helper import MyLMDB, load_class_dictionary


def create_lmdb_by_various_ocr_dataset(lmdb_path, root_folder_path, train_val, category):
    """
     "다양한 형태의 한글 문자 OCR"
    Args:
    Returns:

    """

    db = MyLMDB(lmdb_path)
    ch_class = load_class_dictionary('data_generation/kr_labels.txt')
    for sub_folder_name in os.listdir(root_folder_path):  # 1, 2, 3 ~~
        for file_name in os.listdir(os.path.join(root_folder_path, sub_folder_name)):
            if file_name.split('.')[-1] != 'json':
                continue
            label_file_path = os.path.join(root_folder_path, sub_folder_name, file_name)
            with open(label_file_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            text = ''
            if category == 'word':
                for ch_info in obj['text']['word']:
                    ch = ch_info['value']
                    if ch_class.get(ch) is None:
                        print(f'skip {ch}')
                        text = ''
                        break
                    text += ch
            else:
                ch = obj['info']['text']
                if ch_class.get(ch) is None:
                    print(f'skip {ch}')
                else:
                    text = ch

            if text == '':
                break
            else:
                image_path = label_file_path.replace(f'[라벨]{train_val}_필기체', f'[원천]{train_val}_필기체').\
                    replace('.json', '.jpg')
                db.write_image_label(image_path, text)
    db.close()


def crate_lmdb_by_aihub():
    output_path = 'data_generation/handwritten_sentence'
    train_db = MyLMDB(os.path.join(output_path, 'train'))
    val_db = MyLMDB(os.path.join(output_path, 'val'))
    ch_class = load_class_dictionary('data_generation/kr_labels.txt')
    text_in_wild_data_path = "D:\\Data\OCR\\processed\\aihub\\01.손글씨\\"
    json_path = os.path.join(text_in_wild_data_path, 'handwriting_data_info1.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        json_obj = json.load(f)

    type_dict = {}
    for annotation in json_obj['annotations']:
        text = annotation['text']
        image_id = annotation['image_id']
        image_type = annotation['attributes']['type']
        if not is_valid_text(ch_class, text):
            print(f'skip {text}')
            continue

        image_path = get_image_path(text_in_wild_data_path, image_id)
        if image_path is None:
            print(f'skip {text_in_wild_data_path}')
            continue

        if random.random() < 0.001:
            val_db.write_image_label(image_path, text)
        else:
            train_db.write_image_label(image_path, text)

    train_db.close()
    val_db.close()


def is_valid_text(class_dict, text_data):
    for ch in text_data:
        ch = ch.upper()
        if ch == ' ':
            continue
        if class_dict.get(ch) is None:
            return False
    return True


def get_image_path(root_path, id):
    folder_list = ['1_sentence', '1_syllable', '1_word', '2_sentence', '2_syllable']
    for folder_name in folder_list:
        image_path_candidate = os.path.join(root_path, folder_name, id + '.png')
        if os.path.exists(image_path_candidate):
            return image_path_candidate
    return None


if __name__ == '__main__':
    # output_path = 'data_generation/handwritten_sentence/val'
    # root_path = "D:\\Download\\다양한 형태의 한글 문자 OCR\\Validation\\[라벨]Validation_필기체\\1.글자"
    # train_val_type = 'Validation'
    # data_category = 'character'
    # create_lmdb_by_various_ocr_dataset(output_path, root_path, train_val_type, data_category)
    #
    # output_path = 'data_generation/handwritten_sentence/val'
    # root_path = "D:\\Download\\다양한 형태의 한글 문자 OCR\\Validation\\[라벨]Validation_필기체\\2.단어"
    # train_val_type = 'Validation'
    # data_category = 'word'
    # create_lmdb_by_various_ocr_dataset(output_path, root_path, train_val_type, data_category)

    output_path = 'data_generation/handwritten_sentence/train'
    root_path = "D:\\Download\\다양한 형태의 한글 문자 OCR\\Training\\[라벨]Training_필기체\\1.글자"
    train_val_type = 'Training'
    data_category = 'character'
    create_lmdb_by_various_ocr_dataset(output_path, root_path, train_val_type, data_category)

    output_path = 'data_generation/handwritten_sentence/train'
    root_path = "D:\\Download\\다양한 형태의 한글 문자 OCR\\Training\\[라벨]Training_필기체\\2.단어"
    train_val_type = 'Training'
    data_category = 'word'
    create_lmdb_by_various_ocr_dataset(output_path, root_path, train_val_type, data_category)

    #

    #
    #

    # if type == '문장':
    #     image_folder = ''
    # for ch in text:
    #     if ch_class.get(ch) is None:
    #         break
