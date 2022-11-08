##########################
#  CUSTOM CIFAR DATASETS #
##########################

import torchvision
import numpy as np
from PIL import Image
import math
import os
from mypath import Path

    
class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, ood_noise, id_noise, train=True, transform=None, cont=False, transform_cont=None, consistency=False, download=False, corruption="inet", seed=0):        
        super(CIFAR100, self).__init__(root, train=train, transform=transform, target_transform=None, download=download)
        #CIFAR100red
        self.num_class = 100
        self.train = train
        self.ood_noise = ood_noise
        self.id_noise = id_noise
        self.cont = cont
        self.consistency = consistency
        self.transform_cont = transform_cont
        self.ids_id = []
        self.ids_ood = []
        self.ids_clean = []

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
       
        #Same dataset accross experiments
        seeds = [round(math.exp(1) * 1000), round(math.sqrt(2) * 1000), round(math.pi * 1000)]
        np.random.seed(seeds[seed])
        #OOD noise
        if train and self.ood_noise > 0:
            self.ids_ood = [i for i, t in enumerate(self.targets) if np.random.random() < self.ood_noise]
            if corruption == "inet":
                print('Corrupting CIFAR-100 with ImageNet32 data')
                from datasets.imagenet32 import ImageNet
                imagenet32 = ImageNet(root=Path.db_root_dir("imagenet32"), size=32, train=True)
                ood_images = imagenet32.data[np.random.permutation(np.arange(len(imagenet32)))[:len(self.ids_ood)]]
            elif corruption == "places":
                print('Corrupting CIFAR-100 with Places365 data')
                images_dir = np.array(os.listdir(Path.db_root_dir("places")))
                images_dir.sort()#Important with listdir
                ood_images = images_dir[np.random.permutation(np.arange(len(images_dir)))[:len(self.ids_ood)]]
                ood_images = np.array([np.array(Image.open(os.path.join(Path.db_root_dir("places"), im)).resize((32, 32), resample=2).convert('RGB')) for im in ood_images])#Better could be done
            self.data[self.ids_ood] = ood_images

        #sym ID noise
        if train:
            self.ids_not_ood = [i for i in range(len(self.targets)) if i not in self.ids_ood]
            self.ids_id = [i for i in self.ids_not_ood if np.random.random() < (self.id_noise/(1-self.ood_noise))]
            self.ids_clean = [i for i in self.ids_not_ood if i not in self.ids_id]
            self.targets = np.array([t if i not in self.ids_id else int(np.random.random() * 100) for i, t in enumerate(self.targets)])            
            

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # to return a PIL Image
        image = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image)
            if self.consistency:
                img_ = self.transform(image)
            else:
                img_ = img

        sample = {'image':img, 'image_':img_, 'target':target, 'index':index}
        if self.cont:
            sample['image2'] = self.transform_cont(image)
        return sample
