import os
import sys
import re
import six
import math
import lmdb
import torch
import random
import cv2 
from natsort import natsorted
from PIL import Image, ImageOps
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import pathlib 


def create_gaussian_noise_background(height, width, rgb):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    if rgb:
        return Image.fromarray(image).convert("RGB")    
    else:
        return Image.fromarray(image).convert("L")


def create_quasicrystal_background(height, width, rgb):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = random.random() * 30 + 20  # frequency
    phase = random.random() * 2 * math.pi  # phase
    rotation_count = random.randint(10, 20)  # of rotations

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

    if rgb:
        return image.convert("RGB")    
    else:
        return image


def create_random_picture_background(picture_im_list, height, width):
    im = random.choice(picture_im_list)
    back_w, back_h = im.size 
    if back_w < width or back_h < height: 
        im = im.resize((width, height)) 
    else:
        i = random.randint(0, back_w - width)
        j = random.randint(0, back_h - height)
        im = im.crop((i,j, i + width, j + height))
    return im 


class Batch_Balanced_Dataset(object):

    def __init__(self, opt, mode):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d], mode=mode)
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text, _, _ = data_loader_iter.next()
                # print(f'get_batch: {image.size} {text}')
                balanced_batch_images.append(image)
                # if random.random() < 0.001:
                #     image.save
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text, _, _ = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError as e:
                print(f'get_batch:{e}')
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/', mode='Train'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, mode)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt, mode='Train'):

        self.root = root
        self.opt = opt
        print(root)
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        # 원래 의도는 이미 aug 되어 있는 데이터는 제외하는 것이었는데 aug non augmenated data 가 뒤섞여 있어서 일단 모두 augmentation을 수행해 본다. 
        self.aug = True if self.root.find('data_lmdb_release')>=0 else None
        self.mode = mode 
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3), 
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 5))] 
            )

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)        

        if not self.aug and self.mode== 'Train':
            bg_path = os.path.join(pathlib.Path(self.root).parent.parent, 'background')
            if opt.rgb: 
                self.bg_im_list = [Image.open(os.path.join(bg_path, x)).convert('RGB') for x in os.listdir(bg_path)]               
            else:
                self.bg_im_list = [Image.open(os.path.join(bg_path, x)).convert('L') for x in os.listdir(bg_path)]           
            

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        print(f'The length of the label is longer than max_length: length {len(label)}, {label} in dataset {self.root}')
                        continue

                    if label == '###':
                        continue

                    if label == '***':
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    # out_of_char = f'[^{self.opt.character}]'
                    # out_of_char = '[^' + ''.join(self.opt.character) + ']'
                    # if re.search(out_of_char, label):
                    #     print(f'{label} has out of class')
                    #     continue
                    
                    # out_of_char = f'[{"".join(self.opt.character)}]'
                    # is_not_target_class = False
                    # for char in label:
                    #     match_class = re.search(out_of_char, char)
                    #     if not match_class:
                    #         is_not_target_class = True
                    #         print(f'{label} is not class target, {char} not in class list')
                    #         break
                    # if is_not_target_class:
                    #     continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                original_image = Image.open(buf)
                if self.opt.rgb:
                    img = original_image.convert('RGB')  # for color image
                else:
                    img = original_image.convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.upper()   # by hjpark 우리 class 는 대문자로 되어 있다. 
            if not self.opt.include_space: 
                label = label.replace(' ', '')

            filtered_label = label 
            # # space를 강제로 제거하고 테스트한다. 
            # label = label.replace(' ', '')

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            # out_of_char = f'[^{self.opt.character}]'
            # out_of_char = '[^' + ''.join(self.opt.character) + ']'
            # filtered_label = re.sub(out_of_char, '', label)

            # filtered_label = re.sub('/','',label)

            # if label != filtered_label:
            #     print(f'{label} has out of class data')
            # if random.random() > 0.5:
            #     angle = random.randint(-2, 2)
            #     img = transforms.functional.rotate(img, angle, expand=True)

            if self.mode == 'Train':
                if random.random() > 0.5:
                    angle = random.randint(-3, 3)
                    img = transforms.functional.rotate(img, angle, expand=True, fill=255)

                if not self.aug:
                    w, h = img.size 
                    background_p = random.random() 
                    bg = None 
                    if background_p < 0.1:
                        bg = create_gaussian_noise_background(h, w, rgb=self.opt.rgb)
                    elif background_p < 0.2: 
                        bg = create_quasicrystal_background(h, w, rgb=self.opt.rgb)
                    elif background_p < 0.9:
                        bg = create_random_picture_background(self.bg_im_list, h, w) 
                    if bg is not None:
                        img = Image.blend(img, bg, 0.5)                

                if random.random() < 0.1:                                 
                    img = ImageOps.invert(img)

                if img.size[0] > 3 and img.size[1] > 3:
                    img = self.transform(img)
 
                if random.random() < 0.1: 
                    (w, h) = img.size
                    resized_height = random.choice([12, 13, 14, 15, 16])
                    resized_width = math.ceil(float(resized_height/h) *w)                    
                    img = img.resize((resized_width, resized_height), Image.BICUBIC)

        return (img, filtered_label, original_image, index)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        label = self.image_path_list[index].split('_L_')[1]
        return img, label
        # return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels, original_images, indexes = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))
            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            #max_width = max([math.ceil(self.imgH * image.size[0]/float(image.size[1])) for image in images])
            transform = ResizeNormalize((self.imgW, self.imgH))
 #           transform = ResizeNormalize((max_width, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)


        # if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
        #     resized_max_w = self.imgW
        #     input_channel = 3 if images[0].mode == 'RGB' else 1
        #     transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        #     resized_images = []
        #     for image in images:
        #         w, h = image.size
        #         ratio = w / float(h)
        #         if math.ceil(self.imgH * ratio) > self.imgW:
        #             resized_w = self.imgW
        #         else:
        #             resized_w = math.ceil(self.imgH * ratio)

        #         resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
        #         resized_images.append(transform(resized_image))
        #         # resized_image.save('./image_test/%d_test.jpg' % w)

        #     image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        # else:
        #     transform = ResizeNormalize((self.imgW, self.imgH))
        #     image_tensors = [transform(image) for image in images]
        #     image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels, original_images, indexes


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
