import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from mypath import Path
import json

class MiniImagenet(Dataset):
    def __init__(self, data, labels, transform=None, transform_cont=None, cont=False, consistency=False):
        self.num_class = 100
        self.data, self.targets =  data, labels
        self.transform = transform
        self.transform_cont = transform_cont
        self.cont = cont
        self.consistency = consistency
                
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img)#.convert('RGB')
        
        if self.transform is not None:
            img_t = self.transform(img)
            if self.consistency:
                img_ = self.transform(img)
            else:
                img_ = img_t
                            
        sample = {'image':img_t, 'image_':img_, 'target':target, 'index':index}
        if self.cont:
            sample['image2'] = self.transform_cont(img)            
        return sample

    def __len__(self):
        return len(self.data)


def make_dataset(root=Path.db_root_dir('miniimagenet_preset'), noise_ratio="0.3", noise_type="red"):
    np.random.seed(42)
    nclass = 100
    img_paths = []
    labels = []
    clean_noisy = []
    clean_anno = json.load(open(os.path.join(root, "mini-imagenet-annotations.json")))["data"]
    anno_dict = {}
    for anno in clean_anno:
        anno_dict[anno[0]['image/id']] = int(anno[0]['image/class/label/is_clean'])
    
    for split in ["training", "validation"]:
        if split == "training":
            class_split_path = os.path.join(root, split, '{}_noise_nl_{}'.format(noise_type, noise_ratio))
        else:
            train_num = len(img_paths)
            class_split_path = os.path.join(root, split)
        for c in range(nclass):
            class_img_paths = os.listdir(os.path.join(class_split_path, str(c)))
            class_img_paths.sort()
            for paths in class_img_paths:
                if paths[0] != "n" and split == "training":
                    clean_noisy.append(anno_dict[paths.replace(".jpg","")])
                elif split == "training":
                    clean_noisy.append(1)
                img_paths.append(os.path.join(class_split_path, str(c), paths))
                labels.append(c)

    labels = np.array(labels)
    train_paths = img_paths[:train_num]
    train_labels = labels[:train_num]
    val_paths = img_paths[train_num:]
    val_labels = labels[train_num:]
    
    return train_paths, train_labels, val_paths, val_labels, None, None
