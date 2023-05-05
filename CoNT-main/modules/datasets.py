import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset



class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.front_or_slide_path = args.front_or_slide_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.front_or_slide = json.loads(open(self.front_or_slide_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        # pixel1 = pixel_stat_demo(os.path.join(self.image_dir, image_path[0]))
        # pxiel2 = pixel_stat_demo(os.path.join(self.image_dir, image_path[1]))
        # print(pixel1<pxiel2)
        img1_path = os.path.join(self.image_dir, image_path[0])
        img2_path = os.path.join(self.image_dir, image_path[1])

        image_1 = Image.open(img1_path).convert('RGB')
        image_2 = Image.open(img2_path).convert('RGB')

        image_1_flag = self.front_or_slide[img1_path]
        image_2_flag = self.front_or_slide[img2_path]

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        if int(image_1_flag)==0 and int(image_2_flag)==1:
            image = torch.stack((image_1, image_2), 0)
        elif int(image_1_flag)==1 and int(image_2_flag)==0:
            image = torch.stack((image_2, image_1), 0)
        elif int(image_1_flag)==0 and int(image_2_flag)==0:
            image = torch.stack((image_1, image_2), 0)
            print('error',img1_path,img2_path)
        else:
            image = torch.stack((image_1, image_2), 0)
            print('error',img1_path,img2_path)
        # flag=torch.stack((image_1_flag,image_2_flag),0)
        flag = [image_1_flag, image_2_flag]
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image,  report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
