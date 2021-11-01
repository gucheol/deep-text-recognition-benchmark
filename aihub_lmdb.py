import json
import cv2
import random as rnd
import math
import numpy as np

from data_generation import distorsion_generator


def gaussian_noise(height, width):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    return Image.fromarray(image).convert("RGBA")


def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = rnd.random() * 30 + 20  # frequency
    phase = rnd.random() * 2 * math.pi  # phase
    rotation_count = rnd.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return image.convert("RGBA")


from PIL import Image
fg = Image.open('data_generation/1.png').convert('RGBA')
bg = Image.open('data_generation/background/invoice_background.jpg').convert('RGBA')
w, h = fg.size
# bg = quasicrystal(h, w)
mask = Image.new('RGB', fg.size)
fg.show()
fg, mask = distorsion_generator.sin(fg, mask, horizontal=True)

# fg = fg.rotate(10, expand=True, fillcolor=(255, 255, 255, 255))
bg = bg.resize(fg.size)
blended = Image.blend(fg, bg, 0.5)
blended.show()
#
# # 자 정리를 해보자.
# """
# 1. lmdb session을 하나 연다. 단어+글자 vs sentence로 구분한다.
# 2. data 하나 읽어온다.
# 3. distorsion을 random으로 30%의 데이터에 적용한다.
# 4. 전체의 30% 확률로 skew를 0~5 random하게 준다.
# 5. background는 pure white는 20%, gaussian noise 10%, crystal 10%, Picture 40%로 데이터를 생성한다.
#    -. picture는 bakcground folder안에 있는 것 중에서 random하게 선택한다.
#    -. background
#
# """

# fg_im = cv2.imread('data_generation/1.png')
# bg_im = cv2.imread('data_generation/invoice_background.jpg')
# bg_im = cv2.resize(bg_im, (fg_im.shape[1], fg_im.shape[0]))
# blended = cv2.addWeighted(fg_im, 0.5, bg_im, 0.5, 0)
# cv2.imshow('blended', blended)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def data_audit_1():
    old_class_dictionary = load_class_dictaionry('kr_labels.txt')
    image = "data_generation/invoice_background.jpg"
    ch_class = {}
    path = "D:\\Data\OCR\\processed\\aihub\\01.손글씨\\handwriting_data_info1.json"
    with open(path, 'r', encoding='utf-8') as f:
        json_obj = json.load(f)

    for annotation in json_obj['annotations']:
        text = annotation['text']
        if len(text) == 1:
            continue

        for ch in text:
            if ch_class.get(ch) is not None:
                ch_class[ch] += 1
            else:
                ch_class[ch] = 0

    res = sorted(ch_class.items(), key=(lambda x: x[1]), reverse=True)

    path = "D:\\Data\\OCR\\processed\\aihub\\02.인쇄체\\printed_data_info.json"

    with open(path, 'r', encoding='utf-8') as f:
        json_obj = json.load(f)

    for annotation in json_obj['annotations']:
        text = annotation['text']
        if len(text) == 1:
            continue

        for ch in text:
            if ch_class.get(ch) is not None:
                ch_class[ch] += 1
            else:
                ch_class[ch] = 0

    # res = sorted(ch_class.items(), key=(lambda x: x[1]), reverse=True)
    # # print(res)

    # with open('class_stats.txt', 'w', encoding='utf-8') as f:
    #     for item in res:
    #         if old_class_dictionary.get(item[0]) is None:
    #             print(f'{item[0]} is new character class {item[1]}' )
    #
    #         f.write(f'{item[0]}\t{item[1]}\n')
    #
    # for key, value in old_class_dictionary.items():
    #     if ch_class.get(key) is None:
    #         print(f'{key} has in sufficient data')

# """
# 1. 글자 class 통계가 필요하다.
# 2. 인쇄체와 손글씨로 나눠서 데이터를 만든다.
# 3. 문제는 background 너무 단순하다. composition 이 필요하다.
#
# """

## 다양한 형태의 한글 문자 OCR 데이터 통계 확인

# import os
# import glob
#
# path = "D:\\Download\\다양한 형태의 한글 문자 OCR\\Training\\[라벨]Training_필기체\\2.단어\\"
# words = []
# ch_class = {}
# for folder_name in os.listdir(path):
#     for file_name in os.listdir(os.path.join(path, folder_name)):
#         if file_name.split('.')[-1] != 'json':
#             continue
#
#         with open(os.path.join(path, folder_name, file_name), 'r', encoding='utf-8') as f:
#             obj = json.load(f)
#         word = ''
#         for ch_info in obj['text']['word']:
#             ch = ch_info['value']
#             word += ch
#             if ch_class.get(ch) is None:
#                 ch_class[ch] = 0
#             else:
#                 ch_class[ch] += 1
#         words.append(word)

# with open('data_output.txt', 'w', encoding='utf-8') as f:
#     for word in words:
#         f.write(f'{word}\n')
#
#
#
# for key, value in ch_class:
#     if old_class_dictionary.get(key) is None:
#         print(f'{key} is new character')

# ch_class = {}
#
    # audit_ch = ['멎', '팎', '엷', '맣', '갛', '튿']
    with open('data_output.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        for ch in line:
            # if ch in audit_ch:
            #     print(line)
            #
            if ch_class.get(ch) is not None:
                ch_class[ch] += 1
            else:
                ch_class[ch] = 0

    old_class_dictionary = load_class_dictaionry('kr_labels.txt')
    for key, value in old_class_dictionary.items():
        if ch_class.get(key) is None:
            print(f'{key} has in sufficient data')

    for key, value in ch_class.items():
        if old_class_dictionary.get(key) is None:
            print(f'{key} is new character {value}')

# data_audit_1()