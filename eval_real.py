import os
import numpy as np
import PIL
import string
import argparse

import torch
from torch import tensor
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from collections import OrderedDict

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, LmdbDataset
from model import Model
from torchvision.transforms.functional import to_pil_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor[0].cpu().data, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    print(opt.num_class)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)
    
    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    # model.load_state_dict(copy_state_dict(torch.load(opt.saved_model, map_location=device)))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    # demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_data = LmdbDataset(root=opt.image_folder, opt=opt, mode='Val')  # use RawDataset

    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True, drop_last=True)

    log = open(f'./log_demo_result.txt', 'a')
    # predict
    model.eval()
    fail_count, sample_count = 0, 0
    record_count = 1
    with torch.no_grad():
        for image_tensors, image_path_list, original_images, indexes in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)            
                preds_str = converter.decode(preds_index, length_for_pred)


            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            for image_tensor, gt, pred, pred_max_prob, original_image, lmdb_key in zip(image_tensors, image_path_list, preds_str, preds_max_prob, original_images, indexes):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')                    
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                if pred_max_prob.shape[0] > 0 :
                # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                else:
                    confidence_score = 0.0
                pred = pred.replace('[UNK]', '*')
                compare_gt = "".join(x.upper() for x in gt if x.isalnum())
                compare_pred = "".join(x.upper() for x in pred if x.isalnum())
                # compare_gt = "".join([x.upper() for x in gt])
                # compare_pred = "".join([x.upper() for x in pred])

                if  compare_gt != compare_pred:
                    fail_count+=1
                    print(f'{gt:25s}\t{pred:25s}\tFail\t{confidence_score:0.4f}\t{record_count}\n')
                    im = to_pil_image(image_tensor)
                    try: 
                        im.save(os.path.join('result', f'{lmdb_key}_{compare_pred}_{compare_gt}.jpeg'))                    
                        original_image.save(os.path.join('result', f'{lmdb_key}_{compare_pred}_{compare_gt}.jpg'))                    
                    except Exception as e: 
                        print(f'Error: {e} {lmdb_key}_{compare_pred}_{compare_gt}')
                        exit(1)                        
                else:
                    # print(f'{gt:25s}\t{pred:25s}\tSuccess\t{confidence_score:0.4f}')
                    pass
                sample_count += 1
                record_count += 1
        log.close()
        print (f'total accuracy: {(sample_count-fail_count)/sample_count:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    # parser.add_argument('--image_folder', default='./data/evaluation/valid_card_data/', type=str,
    # parser.add_argument('--image_folder', default='../../9fe_Dataset/val//', type=str,

    parser.add_argument('--image_folder', default='/mnt/b/9fe_Dataset/val/general_app_val', type=str,
    # parser.add_argument('--image_folder', default='./data/val/synth', type=str,
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size')
    # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    # parser.add_argument('--saved_model', default='./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/backup/91.307.pth', help="path to saved_model to evaluation")
    # parser.add_argument('--saved_model', default='./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/devrack.pth',
    # parser.add_argument('--saved_model', default='./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111.98.143/best_accuracy.pth',
    parser.add_argument('--saved_model', default='./saved_models/None-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth',
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=39, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    # parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--sensitive', default=True, help='for sensitive character mode')
    # parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--PAD', default=True, help='whether to keep ratio then pad for image resize')

    """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--Transformation', type=str, default='None', help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')

    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    # parser.add_argument('--data_filtering_off', action='store_true' , help='the size of the LSTM hidden state')
    parser.add_argument('--data_filtering_off', default=0, help='the size of the LSTM hidden state')

    parser.add_argument('--include_space', default=False, help='the size of the LSTM hidden state')
    

    opt = parser.parse_args()

    """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    opt.character = []
    with open('/mnt/b/9fe_Dataset/train/kr_labels.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            opt.character.append(line.split()[1])
        # opt.character.append(' ')
            # todo: space를 추가하려면 이곳에 별도 처리 필요
    print(len(opt.character))

    cudnn.benchmark = True
    cudnn.deterministic = True
    # opt.num_gpu = torch.cuda.device_count()
    opt.num_gpu = 1

    demo(opt)
