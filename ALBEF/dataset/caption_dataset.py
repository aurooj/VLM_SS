import json
import os
import random
import math
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption, scale_0_1
from dataset.selective_sampling import SelectiveSampling


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, num_samples=20):

        #this class has selective sampling feature available, can be used when needed by calling SelectiveSampling.shuffle(batch_size)
        self.selective_sampling = SelectiveSampling(ann_file)

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        #choose only a subset of samples, added to run on small set to make sure everything works fine
        # self.ann = self.ann[:num_samples] 
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.all_groups = []
        self.img_ids = {}   
        
        n = 0
        for ann in tqdm(self.ann):
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        print(f"Dataset size:{len(self.ann)}")
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]

    def shuffle(self, bs=8, , rare_grp_ratio=0.375, batch_shuffle=False):
        #function to prepare minibatches based on selective sampling strategy
        # calls shuffle function from the base class (SelectiveSampling) 
        self.ann = self.selective_sampling.shuffle(bs=bs, rare_grp_ratio=rare_grp_ratio, batch_shuffle=batch_shuffle)
    
        
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, num_samples=20):        
        self.ann = json.load(open(ann_file,'r'))
        #choose only a subset of samples, added to run on small set to make sure everything works fine
        self.ann = self.ann[:num_samples] 
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.img2grp = {}
        self.grp2img = {}

        self.txt2grp = {}
        self.grp2txt = {}

        self.grp_id = []

        txt_id = 0
        for img_id, ann in tqdm(enumerate(self.ann)):
            self.image.append(ann['image'])
            self.grp_id.append(ann['group_id'])
            self.img2txt[img_id] = []

            self.img2grp[img_id] = ann['group_id']
            self.grp2img[ann['group_id']] = img_id

            self.txt2grp[txt_id] = ann['group_id']
            self.grp2txt[ann['group_id']] = txt_id

            # print(ann['caption'])
            if isinstance(ann['caption'], list):

                for i, caption in enumerate(ann['caption']):
                    if isinstance(caption, list):
                        caption = caption[0]
                    self.text.append(pre_caption(caption,self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1
            else:
                caption = ann['caption']
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
        print(f"Dataset size:{len(self.text)}")


                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):    

        #this class has selective sampling feature available, can be used when needed by calling SelectiveSampling.shuffle(batch_size)
        self.selective_sampling = SelectiveSampling(ann_file)


        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.all_groups = []

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in tqdm(enumerate(self.ann)):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')  

        image = self.transform(image)
        image = torch.tensor(scale_0_1(image.numpy()))
        
        # save_image(image, '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/img1.png')
        # print(caption)
        # print(image.shape)        
        return image, caption

    def shuffle(self, bs=8, , rare_grp_ratio=0.375, batch_shuffle=False):
        #function to prepare minibatches based on selective sampling strategy
        # calls shuffle function from the base class (SelectiveSampling) 
        self.ann = self.selective_sampling.shuffle(bs=bs, rare_grp_ratio=rare_grp_ratio, batch_shuffle=batch_shuffle)
