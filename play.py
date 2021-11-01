# def get_gt_from_file_name(file_name): 
#     return file_name.split('_')[0]

# print(get_gt_from_file_name('갈김경남_8449541.jpg'))

# import glob

# for path in glob.iglob('/home/hjpark/**', recursive=True):
#     print(path)
import lmdb
from PIL import Image 
import six

env_train = lmdb.open('/home/hjpark/OCR/lmdb/out/val', map_size=101919)
black_list = [] 

with open('/home/hjpark/OCR/lmdb/kr_classes.txt', encoding='ms949') as f:
    while True:
        line = f.readline()
        if not line : break 
        character_class, character_count = line.strip('\n').split('\t')
        if int(character_count) < 1200:
            black_list.append(character_class) 


with env_train.begin(write=True) as txn :
    num_samples = txn.get('num-samples'.encode())
    if num_samples is None:
        exit(1)
    num_samples = int(num_samples)

    for id in range(1, num_samples):
        image_key = 'image-%09d'.encode() % id
        label_key = 'label-%09d'.encode() % id
        label = txn.get(label_key)
        if label is None:
            print ('label %s is null' % str(label_key))
            exit (1)
        label = label.decode('utf-8')
        new_label = ''
        for ch in label :
            if ch in black_list:
                new_label += '<UNK>'
            else:
                new_label += ch 

        if id % 10000 == 0:
            image = txn.get(image_key)
            buf = six.BytesIO()
            buf.write(image)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')  # for color image
#            image = Image.frombytes(image)
            img.save('jpg/' + new_label + '.jpg')

#cache = {} 
#cache['a'.encode()] = bytes([3])
#cache['b'.encode()] = bytes([4])

#with env_train.begin(write=True) as txn:
#    for k, v in cache.items():
#        txn.put(k, v)

#with env_train.begin(write=True) as txn :
#    a = txn.get('a'.encode())
#    a = int.from_bytes(a, byteorder='big', signed=True)
#    print(a)
