import os 
import io
import random
from lmdb_helper import MyLMDB, load_class_dictionary, is_valid_label, image_bin_to_pil


if __name__== '__main__':
    target = '../data/tr/train/handwritten'
    class_dict = load_class_dictionary('../data/tr/train/kr_labels.txt', add_space=True)
    target_db = MyLMDB(target, mode='r', map_size=6e9)

    scale_dict = {}
    num_of_samples = target_db.num_of_samples
    for i in range(1, num_of_samples):
        im, label = target_db.read_image_label(i)

        pil_im = image_bin_to_pil(im)

        w, h = pil_im.size
        scaled_w = int(w/h * 32)
        if scale_dict.get(scaled_w) is not None:
            scale_dict[scaled_w] += 1
        else:
            scale_dict[scaled_w] = 1
    
    sorted_list = sorted(scale_dict.items())
    for item in sorted_list:
        print(f'{item[0]}, {item[1]}')
