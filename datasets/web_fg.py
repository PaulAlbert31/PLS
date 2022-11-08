from torch.utils.data import Dataset
from PIL import Image
from mypath import Path
import numpy as np
import torch
import os

class fg_web_dataset(Dataset): 
    def __init__(self, transform, mode, transform_cont=None, cont=False, consistency=False, which='web-bird'):
        classes_n = {'web-bird': 200, 'web-car':196, 'web-aircraft':100}
        self.num_class = classes_n[which]
        self.root = Path.db_root_dir(which)
        self.transform = transform
        self.mode = mode
        self.transform_cont = transform_cont
        self.cont = cont
        self.consistency = consistency
        
        if self.mode=='test':
            with open(os.path.join(self.root, 'val-list.txt')) as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = []
            for line in lines:
                img, target = line.split(':')
                target = int(target)                
                self.val_imgs.append(os.path.join(self.root, img))
                self.val_labels.append(target)
        else:    
            with open(os.path.join(self.root, 'train-list.txt')) as f:
                lines=f.readlines()    
            train_imgs = []
            self.targets = []
            for line in lines:
                img, target = line.split(':')
                target = int(target)
                train_imgs.append(os.path.join(self.root, img))
                self.targets.append(target)
                    
            self.data = train_imgs
            self.targets = np.array(self.targets)
                            
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.data[index]
            target = self.targets[index]
            
            img = Image.open(img_path)
            
            if self.transform is not None:
                img_t = self.transform(img)
                
            if self.consistency:
                img_ = self.transform(img)
            else:
                img_ = img_t
                            
            sample = {'image':img_t, 'image_':img_, 'target':target, 'index':index}
            if self.cont:
                sample['image1'] = self.transform_cont(img)
                sample['image2'] = self.transform_cont(img)
            return sample
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
                       
            image = Image.open(img_path).convert('RGB')
            
            img = self.transform(image) 
            return {'image':img, 'target':target, 'index':index}
           
    def __len__(self):
        if self.mode!='test':
            return len(self.data)
        else:
            return len(self.val_imgs)    
