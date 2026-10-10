import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as tfs
import cv2
import pandas as pd
from .utils import GetTransforms, transform
import numpy as np
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.image_dir = os.path.join(args.data_dir, args.dataset_name,  'images')
        self.ann_path = os.path.join(args.data_dir, args.dataset_name, 'annotation.json')
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.labels_path = os.path.join(args.data_dir, args.dataset_name, args.label_path)
        self.labels = json.loads(open(self.labels_path, 'r').read())

        self.examples = self.ann[self.split]
        if args.dataset_name == 'iu_xray':
            self._labels = []
            for e in self.examples:
                img_id = e['id']
                array = img_id.split('-')
                modified_id = array[0] + '-' + array[1]
                self._labels.append(self.labels[modified_id])
        elif args.dataset_name =='mimic_cxr_dsr2':
            self._labels = [self.labels[e['id']] for e in self.examples]

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        array = image_id.split('-')
        modified_id = array[0] + '-' + array[1]
        label = np.array(self.labels[modified_id]).astype(np.float32)
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        label = np.array(self.labels[image_id]).astype(np.float32)
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length,label)
        return sample


class ChexPert(Dataset):
    def __init__(self, label_path, cfg, mode='train', transform = None):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        self.transform = transform
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in tqdm(f):
                labels = []
                fields = line.strip('\n').split(',')
                #print('111', fields)
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    else:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                image_path = os.path.join('./data', image_path)
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


class IuxrayMultiImageClsDataset(Dataset):
    def __init__(self, args, split, transform=None, vis =False):
        self.image_dir = os.path.join(args.data_dir, args.dataset_name,  'images')
        self.ann_path = os.path.join(args.data_dir, args.dataset_name, 'annotation.json')
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.labels_path = os.path.join(args.data_dir, args.dataset_name, args.label_path)
        self.labels = json.loads(open(self.labels_path, 'r').read())
        self.transform = transform
        self._labels = []
        for e in self.examples:
            img_id = e['id']
            array = img_id.split('-')
            modified_id = array[0] + '-' + array[1]
            self._labels.append(self.labels[modified_id])
        self.vis = vis


    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        array = image_id.split('-')
        modified_id = array[0] + '-' + array[1]
        label = np.array(self.labels[modified_id]).astype(np.float32)
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)

        if self.vis:
            return image, image_path, label
        else:
            return image, label, modified_id

    def __len__(self):
        return len(self.examples)


class MimiccxrSingleImageClsDataset(BaseDataset):
    def __init__(self, args, split, transform=None, vis = False):
        self.image_dir = os.path.join(args.data_dir, args.dataset_name, 'images')
        self.ann_path = os.path.join(args.data_dir, args.dataset_name, 'annotation.json')
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.labels_path = os.path.join(args.data_dir, args.dataset_name, args.label_path)
        self.labels = json.loads(open(self.labels_path, 'r').read())
        self.transform = transform
        self._labels = [self.labels[e['id']] for e in self.examples]
        self.vis = vis

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        label = np.array(self.labels[image_id]).astype(np.float32)
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.vis:
            return image, image_path, label
        else:
            return image, label, image_id

    def __len__(self):
        return len(self.examples)

