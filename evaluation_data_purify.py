import os
import glob
import shutil
from PIL import Image

# for file_path in glob.glob('./data/evaluation/card_evaluation/*.png'):
#     file_name = os.path.basename(file_path)
#     if file_name.find(',') >= 0 or file_name.find('@') >=0 or file_name.find('*') > 0 or file_name.find(':') > 0 \
#             or file_name.find('r') > 0 or file_name.find('성별') > 0 or file_name.find('KOR') > 0 or file_name.find('~') > 0:
#         # shutil.move(file_path, os.path.join('result', 'no_way_out', file_name))
#         # print(file_path, os.path.join('result', 'no_way_out', file_name))
#         continue
#     shutil.copy(file_path, os.path.join('./data/evaluation/valid_card_data', file_name))


src = './data/evaluation/valid_card_data/'
# target = './data/evaluation/deleted/'
# for file_path in glob.glob('./result/tagging_error/*.png'):
#     file_name = os.path.basename(file_path)
#     src_file_path = os.path.join(src, file_name)
#     print(src_file_path, os.path.join(target, file_name))
#     shutil.move(src_file_path, os.path.join(target, file_name))

for file_path in glob.glob(src + '*.png'):
    base_name = os.path.basename(file_path)
    if file_path.find('_(F-4)_') > 0:
        target_file_path = file_path.replace('_(F-4)_', '_재외동포(F-4)_')
        shutil.move(file_path, target_file_path)
        # print(file_path, target_file_path)
        # continue
    # if base_name.find('e') > 0 :
    #     print(file_path)
