""" Detection dataset

Hacked together by Ross Wightman
"""
import random
import os
import torch.utils.data as data
import numpy as np
import json
from PIL import Image
import torch
from .parsers import create_parser


class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None,split=None,data_root=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir


        self.task_dir=os.path.join(data_root,'task.json')
        with open(self.task_dir, 'r') as f:
            task_json = json.load(f)

        self.split=split

        self.task=task_json[split]

        self.prompt=None

        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser

        self.cat_id_to_label = {cat_id: i+5  for i, cat_id in enumerate(self._parser.cat_ids)}

        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        #here choose a task
        if self.split=='train':
            task=random.choice(self.task)
        elif self.split=='val':
            task=self.prompt

        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index,task)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        if task[0]==-1:
             mask=[self.cat_id_to_label[task[1]]]#task[0]
        elif task[-1]==-1:
             mask=[*task[0:-1]]#task[0]
        else:
            mask = [*task[0:-1], self.cat_id_to_label[task[1]]]
        
        #prompt=np.array(mask)

        prompt=torch.zeros(9).scatter_(0, torch.tensor(mask), 1).numpy()#np.array(mask)#

        return img, target, prompt

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class SkipSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        n (int): skip rate (select every nth)
    """
    def __init__(self, dataset, n=2):
        self.dataset = dataset
        assert n >= 1
        self.indices = np.arange(len(dataset))[::n]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def parser(self):
        return self.dataset.parser

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        self.dataset.transform = t
