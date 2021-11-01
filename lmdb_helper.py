import os
import io
import lmdb
import six
from PIL import Image

IMAGE_SAMPLE_HEIGHT = 64

def image_bin_to_pil(image_bin):
    buf = six.BytesIO()
    buf.write(image_bin)
    buf.seek(0)
    img = Image.open(buf)
    return img
 

def is_valid_label(label, classes):
    for ch in label:
        if classes.get(ch) is None:
            print(f'{ch} is not valid')
            return False
    return True


def load_class_dictionary(path, add_space=False):
    class_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        items = line.strip().split()
        class_dict[items[1]] = 0

    if add_space:
        class_dict[' '] = 0 

    return class_dict


def load_and_resize(path, resize=False):
    im = Image.open(path)
    w, h = im.size
    if h > w * 2:
        print(f'Warning: {path} w={w} h={h} ')
    if resize:
        scaled_w = int(IMAGE_SAMPLE_HEIGHT / h * w)
        im = im.resize((scaled_w, IMAGE_SAMPLE_HEIGHT), Image.LANCZOS)

    with io.BytesIO() as output:
        im.save(output, format="JPEG")
        contents = output.getvalue()
    return contents


class MyLMDB:
    def __init__(self, path, mode='w', sync_period=500, map_size=1e10):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.session = lmdb.open(path, map_size=map_size)
        self.mode = mode
        self.cache = {}
        self.sync_period = sync_period
        self.num_of_write = 0
        self.num_of_samples = self.get_number_of_samples()

    def get_number_of_samples(self):
        with self.session.begin(write=False) as txn:
            num_samples = txn.get('num-samples'.encode())
            if num_samples is None:
                num_of_samples = 1
            else:
                num_of_samples = int(num_samples)
        return num_of_samples

    def write_im_label(self, im, label):        
        if self.mode != 'w':
            print('This db is opened as ReadOnly mode')
            exit(-1)

        image_key = 'image-%09d'.encode() % self.num_of_samples
        label_key = 'label-%09d'.encode() % self.num_of_samples
        self.cache[image_key] = im
        self.cache[label_key] = label.encode()

        self.num_of_samples += 1
        self.num_of_write += 1
        if self.num_of_write > self.sync_period:
            print(f'{self.path} cache write {self.num_of_samples}')
            self.cache['num-samples'.encode()] = str(self.num_of_samples - 1).encode()
            with self.session.begin(write=True) as txn:
                for k, v in self.cache.items():
                    txn.put(k, v)
            self.num_of_write = 0
            self.cache = {}

    def write_image_label(self, image_path, label, resize=False):        
        if self.mode != 'w':
            print('This db is opened as ReadOnly mode')
            exit(-1)

        image_bin = load_and_resize(image_path, resize)
        image_key = 'image-%09d'.encode() % self.num_of_samples
        label_key = 'label-%09d'.encode() % self.num_of_samples
        self.cache[image_key] = image_bin
        self.cache[label_key] = label.encode()

        self.num_of_samples += 1
        self.num_of_write += 1
        if self.num_of_write > self.sync_period:
            print(f'{self.path} cache write {self.num_of_samples}')
            self.cache['num-samples'.encode()] = str(self.num_of_samples - 1).encode()
            with self.session.begin(write=True) as txn:
                for k, v in self.cache.items():
                    txn.put(k, v)
            self.num_of_write = 0
            self.cache = {}

    def read_image_label(self, index):
        label_key = 'label-%09d'.encode() % index
        image_key = 'image-%09d'.encode() % index

        with self.session.begin(write=False) as txn:
            im = txn.get(image_key)
            label = txn.get(label_key).decode('utf-8')            
            return im, label

    def close(self):
        if self.mode != 'w':
            return 
        self.cache['num-samples'.encode()] = str(self.num_of_samples - 1).encode()
        with self.session.begin(write=True) as txn:
            for k, v in self.cache.items():
                txn.put(k, v)
