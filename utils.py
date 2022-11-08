import torch
import torchvision
import os
import numpy as np
import datasets
import torch.nn.functional as F
import copy
import PIL
import math
from mypath import Path


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine_lr(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs

    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))

    return lrs

def min_max(x):
    return (x - x.min())/(x.max() - x.min())

def multi_class_loss(pred, target):
    pred = F.softmax(pred, dim=1)
    loss = - torch.sum(target*torch.log(pred), dim=1)
    return loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def make_data_loader(args, no_aug=False, transform=None, **kwargs):
    if args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        size1 = 32
        size = 32
    elif args.dataset == 'miniimagenet_preset':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]
        size1 = 32
        size = 32
    elif 'web-' in args.dataset:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size1 = 512
        size = 448
    elif args.dataset == 'custom':
        #Replace by the values for your dataset
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size1 = 256
        size = 224
    

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size1, interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.RandomCrop(size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    
    transform_cont = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.RandomResizedCrop(size, interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    if args.dataset == "cifar100":
        from datasets.cifar import CIFAR100
        trainset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_noise, id_noise=args.id_noise, train=True, transform=transform_train, transform_cont=transform_cont, cont=args.cont, consistency=True, seed=args.seed, corruption=args.corruption)
        trackset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_noise, id_noise=args.id_noise, train=True, transform=transform_test, transform_cont=transform_train, cont=False, consistency=False, seed=args.seed, corruption=args.corruption)
        trackset.data, trackset.targets = trainset.data, trainset.targets
        testset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_noise, id_noise=args.id_noise, train=False, transform=transform_test)
    elif args.dataset == "miniimagenet_preset":
        from datasets.miniimagenet_preset import make_dataset, MiniImagenet
        train_data, train_labels, val_data, val_labels, test_data, test_labels = make_dataset(noise_ratio=args.noise_ratio)
        trainset = MiniImagenet(train_data, train_labels, transform=transform_train, transform_cont=transform_cont, cont=args.cont, consistency=True)
        trackset = MiniImagenet(train_data, train_labels, transform=transform_test, transform_cont=transform_train, cont=False, consistency=False)
        testset = MiniImagenet(val_data, val_labels, transform=transform_test)
    elif "web-" in args.dataset:
        from datasets.web_fg import fg_web_dataset
        trainset = fg_web_dataset(transform=transform_train, mode="train", transform_cont=transform_cont, cont=args.cont, consistency=True, which=args.dataset)
        trackset = fg_web_dataset(transform=transform_test, mode="train", transform_cont=transform_train, cont=False, consistency=False, which=args.dataset)
        testset = fg_web_dataset(transform=transform_test, mode="test", which=args.dataset)
    elif args.dataset == "custom":
        from datasets.custom import make_dataset, Custom
        train_data, train_labels, val_data, val_labels, test_data, test_labels = make_dataset()
        trainset = Custom(train_data, train_labels, transform=transform_train, transform_cont=transform_cont, cont=args.cont, consistency=True)
        trackset = Custom(train_data, train_labels, transform=transform_test, transform_cont=transform_train, cont=False, consistency=False)
        testset = Custom(val_data, val_labels, transform=transform_test)        
    else:
        raise NotImplementedError("Dataset {} is not implemented".format(args.dataset))
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs) #Normal training
    track_loader = torch.utils.data.DataLoader(trackset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader, track_loader

#Should be improved
def create_save_folder(args):
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed)))
    except:
        pass
