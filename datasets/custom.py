from PIL import Image
from torch.utils.data import Dataset
from mypath import Path
import os
import numpy as np

class Custom(Dataset):
    def __init__(self, data, labels, transform=None, transform_cont=None, cont=False, consistency=False):
        self.num_class = 100
        self.data, self.targets =  data, labels
        self.transform = transform
        self.transform_cont = transform_cont
        self.cont = cont
        self.consistency = consistency
                
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img)
        
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

def make_dataset(root=Path.db_root_dir('custom')):    
    #To edit. *_paths are list of absolute paths to images and *_labels are lists of ints with the respective labels    
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
    #Untested example with a structure in root/{train,val}/{class1,...}/xx.{jpeg,png}
    classes = os.listdir(os.path.join(root, 'train')).sort()
    for split in ['train', 'val']:
        for i, c in enumerate(classes):
            images = os.listdir(os.path.join(root, f'{split}/{c}'))
            for im in images:
                if im.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:continue
                if split == 'train':
                    train_paths.append(os.path.join(root, f'{split}/{c}', im))
                    train_labels.append(i)
                else:
                    val_paths.append(os.path.join(root, f'{split}/{c}', im))
                    val_labels.append(i)       

    train_labels = np.array(train_labels) #Might not be required
    val_labels = np.array(val_labels)
    return train_paths, train_labels, val_paths, val_labels, None, None
