import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import make_data_loader, create_save_folder, multi_class_loss, mixup_data, min_max, cosine_lr, adjust_lr
import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy('file_system')
import copy
from torch.cuda.amp import GradScaler, autocast
from sklearn.mixture import GaussianMixture
import random



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0        
               
        self.kwargs = {'num_workers': 12, 'pin_memory': True}
        
        self.train_loader, self.val_loader, self.track_loader = make_data_loader(args, **self.kwargs)

        self.args.num_class = self.train_loader.dataset.num_class

        if args.net == 'preresnet18':
            from nets.preresnet import PreActResNet18
            model = PreActResNet18(num_classes=self.args.num_class, proj_size=self.args.proj_size) 
        elif args.net == 'resnet50':
            from nets.resnet import ResNet50
            model = ResNet50(num_classes=self.args.num_class, pretrained=True, proj_size=self.args.proj_size)
        else:
            raise NotImplementedError("Network {} is not implemented".format(args.net))
        
        print('Number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

        self.model = nn.DataParallel(model).cuda()
        
        wd = 5e-4
        if 'web-' in self.args.dataset:
            wd = 1e-3
        
        self.optimizer = torch.optim._multi_tensor.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=wd)
            
        self.criterion_nored = nn.CrossEntropyLoss(reduction='none')
        
        if self.args.lr_type == "cosine":
            self.cosine_lr_per_epoch = cosine_lr(self.args.lr, self.args.warmup, self.args.epochs)
            

        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []

        self.scaler = GradScaler(enabled=not self.args.fp32)

        #Init
        self.is_clean = torch.ones(len(self.train_loader.dataset), dtype=torch.bool)
        self.is_idn = torch.zeros(len(self.train_loader.dataset), dtype=torch.bool)
        self.is_ood = torch.zeros(len(self.train_loader.dataset), dtype=torch.bool)
        
        self.weights = torch.ones(len(self.train_loader.dataset))
        
        self.guessed_labels_soft = torch.zeros((len(self.train_loader.dataset), self.args.num_class))
                
    def train(self, epoch):
        self.model.train()

        if self.args.lr_type == "cosine":
            adjust_lr(self.optimizer, self.cosine_lr_per_epoch[epoch])
        acc = 0
        tbar = tqdm(self.train_loader)
        
        track_agreement = torch.zeros(len(self.train_loader.dataset))
        m_dists = torch.tensor([])
        l = torch.tensor([])
        self.epoch = epoch
        total_sum = 0
        for i, sample in enumerate(tbar):
            ids = sample['index']
            im, im_, target = sample['image'].cuda(), sample['image_'].cuda(), sample['target'].cuda()

            weights = torch.ones(len(im))
            if epoch >= self.args.warmup:
                batch_clean = self.is_clean[ids]
                batch_idn = self.is_idn[ids]
                batch_ood = self.is_ood[ids]
                
                with torch.no_grad():
                    with autocast(enabled = not self.args.fp32):
                        out1 = F.softmax(self.model(im), dim=1)
                        out2 = F.softmax(self.model(im_), dim=1)
                        
                        #Label guessing for ID noisy samples
                        guessed_targets = (out1 + out2) / 2                        
                            
                        guessed_targets = guessed_targets ** (self.args.gamma) #temp sharp                        
                        
                        guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True) #normalization
                        guessed_targets = guessed_targets.detach()                        
                                                    
                        self.guessed_labels_soft[ids] = guessed_targets.cpu() #track for the pseudo-loss
                        
                        target[batch_idn] = guessed_targets[batch_idn].cuda()
                        
                        weights[batch_idn] = self.weights[ids[batch_idn]]                        
                                                   
            if self.args.mixup:
                image, la, lb, lam, o = mixup_data(im, target)
                target = lam*target + (1-lam)*target[o]
            else:
                image = im
                
            with autocast(enabled = not self.args.fp32):
                outputs, feats1_cont = self.model(image, return_features=True)
                weights = weights.cuda()
                if epoch < self.args.warmup:
                    loss_c = multi_class_loss(outputs, target)
                else:
                    loss_c = (multi_class_loss(outputs, target) * weights).sum() / weights.sum()
                    
                if not self.args.mixup and epoch >= self.args.warmup:
                    preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                    acc += torch.sum(preds == torch.argmax(target, dim=1))
                    
                total_sum += image.shape[0]
                    
                # class balanced regularization 
                if (epoch >= self.args.warmup and not self.args.no_reg) or self.args.force_reg:
                    prior = torch.ones(self.args.num_class)/self.args.num_class
                    prior = prior.cuda()        
                    pred_mean = torch.softmax(outputs, dim=1).mean(0)
                    penalty = torch.sum(prior*torch.log(prior/pred_mean))
                    loss_c += penalty
                    
                if (self.args.cont and epoch >= self.args.cont_reg) or self.args.no_correct:
                    ii = sample["index"]
                    image2 = sample["image2"].cuda() #Strongly augmented image (SimCLR augs)
                    
                    _, feats2_cont = self.model(image2, return_features=True)
                    
                    feats1_cont, feats2_cont = F.normalize(feats1_cont, p=2), F.normalize(feats2_cont, p=2)
                    logits = torch.div(torch.matmul(feats1_cont, feats2_cont.t()), self.args.mu) #Contrastive temperature
                    
                    #corrected target                    
                    labels = torch.zeros((len(image2), len(image2)+self.args.num_class)).cuda()
                    target_cont = F.one_hot(torch.argmax(target, dim=1), num_classes=self.args.num_class)
                    
                    if self.args.no_weights_cont:
                        weights = torch.ones(len(weights)).cuda().float()
                        
                    labels[:, :self.args.num_class] = weights.view(-1,1) * target_cont
                    labels[:, self.args.num_class:] = (1-weights.view(-1,1)) * torch.eye(len(labels)).cuda()
                       
                    labels = torch.matmul(labels, labels.t())
                    if self.args.mixup and epoch >= self.args.warmup:
                        labels = lam * labels + (1-lam) * labels[o]
                        
                    loss_u = self.args.weight_unsup * multi_class_loss(logits, labels) / labels.sum(dim=-1)                  

                    loss_c += loss_u.mean()                        
                else:
                    loss_u = torch.cuda.FloatTensor([0])                   
                    
            loss = loss_c.mean()
                            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if i % 10 == 0:
                tbar.set_description('Training loss {0:.2f}, LR {1:.6f}, L_class {2:.2f}, L_unsup {3:.2f}'.format(loss.mean(), self.optimizer.param_groups[0]['lr'], loss_c.mean(), loss_u.mean()))

        if epoch == self.args.warmup - 1:
            self.save_model(epoch, t=True)
        #Checkpoint
        self.save_model(epoch)
        print('[Epoch: {}, numImages: {}, numClasses: {}]'.format(epoch, total_sum, self.args.num_class))
        if not self.args.mixup:
            print('Training Accuracy: {0:.4f}'.format(float(acc)/total_sum))
        return
                    
    def val(self, epoch, f=None, dataset='val', save=True):
        self.model.eval()
        acc, acc_ens = 0, 0

        vbar = tqdm(self.val_loader)
        total = 0
        losses, accs = torch.tensor([]), torch.tensor([])
        acc_c = torch.zeros(len(self.val_loader.dataset))
        trainLabels = torch.LongTensor(torch.argmax(self.track_loader.dataset.targets, dim=1))

        #Computing linear val acc & kNN acc
        with torch.no_grad():
            #KNN hyperparam
            K = 200
            C = self.args.num_class + 1
            sigma=.1
            retrieval_one_hot = torch.zeros(K, C)
            top1, top5 = 0, 0
            retrieval_one_hot = retrieval_one_hot.cuda()
            for i, sample in enumerate(vbar):
                image, target, ids = sample['image'], sample['target'], sample['index']
                image, target = image.cuda(), target.cuda()
                    
                with autocast(enabled = not self.args.fp32):
                    outputs, feat_cont = self.model(image, return_features=True)                    
                    loss = self.criterion_nored(outputs, target)
                
                losses = torch.cat((losses, loss.cpu()))

                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                accs = torch.cat((accs, (preds==target.data).float().cpu()))
                
                acc_c[ids] = (preds == target).float().cpu()

                acc += torch.sum(preds == target.data)
                total += preds.size(0)
                
                if f is not None:
                    # KNN with cosine similarity
                    batchSize = image.size(0)
                    features = F.normalize(feat_cont, p=2).float()
                    f = f.cuda()
                    dist = torch.mm(features, f.t())
                    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                    candidates = trainLabels.view(1,-1).expand(batchSize, -1).cuda()
                    retrieval = torch.gather(candidates, 1, yi)
                    
                    retrieval_one_hot.resize_(batchSize * K, C).zero_()
                    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                    yd_transform = yd.clone().div_(sigma).exp_()
                    probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                                yd_transform.view(batchSize, -1, 1)), 1)
                    _, predictions = probs.sort(1, True)
                    
                    # Find which predictions match the target
                    
                    correct = predictions.eq(target.data.view(-1,1))
                    
                    top1 = top1 + correct.narrow(1,0,1).sum().item()
                    top5 = top5 + correct.narrow(1,0,5).sum().item()
                    
                    
                if i % 10 == 0:
                    vbar.set_description('Validation loss: {0:.2f}'.format(loss.mean()))
                    
        final_acc = float(acc)/total
        if i % 10 == 0:
            print('[Epoch: {}, numImages: {}]'.format(epoch, (len(self.val_loader)-1)*self.args.batch_size + image.shape[0]))
        self.acc.append(final_acc)
        torch.save(torch.tensor(self.acc), os.path.join(self.args.save_dir, '{0}_acc.pth.tar'.format(self.args.checkname)))
        if final_acc > self.best and save:
            self.best = final_acc
            self.best_epoch = epoch
            self.save_model(epoch, best=True)
        print('Validation Accuracy: {0:.4f}, ensemble {1:.4f}, kNN {2:.4f}, best accuracy {3:.4f} at epoch {4}'.format(final_acc, float(acc_ens)/total, float(top1)/total, self.best, self.best_epoch))
        return 

    def save_model(self, epoch, t=False, best=False):
        if t:
            checkname = os.path.join(self.args.save_dir, '{}_{}.pth.tar'.format(self.args.checkname, epoch))
        elif best:
            checkname = os.path.join(self.args.save_dir, '{}_best.pth.tar'.format(self.args.checkname, epoch))
            with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                f.write(str(self.best))
        else:
            checkname = os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname, epoch))
            
        torch.save({
            'epoch': epoch+1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best': self.best,
            'best_epoch':self.best_epoch
        }, checkname)
        

    def track_loss(self):
        model = self.model
        model.eval()
        
        acc = 0
        total_sum = 0
        display_acc = 0
        with torch.no_grad():
            tbar = tqdm(self.track_loader)
            tbar.set_description('Tracking loss')

            losses = torch.zeros(len(self.track_loader.dataset))
            features_cont = torch.zeros(len(self.track_loader.dataset), self.args.proj_size)
            pseudo_loss = torch.zeros(len(self.track_loader.dataset))
            
            for i, sample in enumerate(tbar):
                image, target, ids = sample['image'], sample['target'], sample['index']
                target, image = target.cuda(), image.cuda()
                    
                outputs, feats_cont = model(image, return_features=True)
                    
                #Track features
                features_cont[ids] = feats_cont.detach().cpu().float()
                losses[ids] = self.criterion_nored(outputs, torch.argmax(target, dim=-1)).detach().cpu().float()
                               
                pseudo_guess = self.guessed_labels_soft[ids].cuda()
                pseudo_loss[ids] = multi_class_loss(outputs, pseudo_guess).cpu()#self.criterion_nored(outputs, torch.argmax(pseudo_guess, dim=1)).detach().cpu().float()

                preds = F.softmax(outputs, dim=1)
                topk = torch.topk(preds, k=2, dim=1)[0].detach().cpu()
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

                #Track train accuracy
                if self.args.mixup:
                    target = torch.argmax(target, dim=1)
                    display_acc += (preds == target).sum()
                    total_sum += preds.size(0)              
               
            if self.args.mixup:
                print('Training Accuracy: {0:.4f}'.format(float(display_acc)/total_sum))
            
            return F.normalize(features_cont, p=2), losses, pseudo_loss
        
                               
def main():


    parser = argparse.ArgumentParser(description="PyTorch noisy labels PLS")
    parser.add_argument('--net', type=str, default='preresnet18',
                        choices=['resnet50', 'preresnet18'],
                        help='net name (default: preresnet18)')
    parser.add_argument('--dataset', type=str, default='miniimagenet_preset', choices=['miniimagenet_preset', 'cifar100', 'web-brid', 'web-car', 'web-aircraft'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-type', type=str, default="cosine", choices=["step", "cosine"])
    parser.add_argument('--gamma', type=float, default=2, help='Consistency reg temperature')
    parser.add_argument('--mu', type=float, default=0.2, help='Contrastive temperature')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=None, nargs='+', help='Epochs when to reduce lr')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mixup', default=False, action='store_true')

    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--cont', default=False, action='store_true')
    parser.add_argument('--fp32', default=False, action='store_true')
    parser.add_argument('--warmup', default=30, type=int)
    parser.add_argument('--proj-size', type=int, default=128)

    #CIFAR100
    parser.add_argument('--ood-noise', default=.0, type=float)
    parser.add_argument('--id-noise', default=.0, type=float)
    parser.add_argument('--corruption', default="inet", type=str, choices=["inet", "places"])

    #Mini
    parser.add_argument('--noise-ratio', default="0.0", type=str)
    
    #Abla
    parser.add_argument('--no-reg', default=False, action='store_true')
    parser.add_argument('--no-weights', default=False, action='store_true')
    parser.add_argument('--no-correct', default=False, action='store_true')
    parser.add_argument('--thresh', default=.95, type=float)    
    parser.add_argument('--cont-reg', default=None, type=int)
    parser.add_argument('--force-reg', default=False, action='store_true')
    parser.add_argument('--no-weights-cont', default=False, action='store_true')
    parser.add_argument('--weight-unsup', default=1, type=float)
    args = parser.parse_args()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    seeds = {'1': round(torch.exp(torch.ones(1)).item()*1e6), '2': round(torch.acos(torch.zeros(1)).item() * 2), '3':round(torch.sqrt(torch.tensor(2.)).item()*1e6)}
    try:
        torch.manual_seed(seeds[str(args.seed)])
        torch.cuda.manual_seed_all(seeds[str(args.seed)])  # GPU seed
        random.seed(seeds[str(args.seed)])  # python seed for image transformation                                                                                                                                                                                              
    except:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
        random.seed(args.seed)
        
    if args.steps is None:
        args.steps = [args.epochs]
    if args.cont_reg is None:
        args.cont_reg = args.warmup
        
    create_save_folder(args)
    args.checkname = args.net + '_' + args.dataset
    args.save_dir = os.path.join(args.save_dir, args.checkname, args.exp_name, str(args.seed))    
    
    _trainer = Trainer(args)

    #One hot labels for all
    relabel = torch.tensor(_trainer.train_loader.dataset.targets)
    relabel = F.one_hot(relabel, num_classes=args.num_class).float()    
    _trainer.train_loader.dataset.targets = relabel
    _trainer.track_loader.dataset.targets = relabel
    _trainer.guessed_labels_soft = relabel.clone()
    
    start_ep = 0
    if args.resume is not None:
        load_dict = torch.load(args.resume, map_location='cpu')
        _trainer.model.module.load_state_dict(load_dict['state_dict'])
        _trainer.optimizer.load_state_dict(load_dict['optimizer'])        
        start_ep = load_dict['epoch']
        del load_dict
        v = _trainer.val(start_ep)
        
 
    targets = torch.argmax(_trainer.train_loader.dataset.targets, dim=-1)
    preds = torch.ones(len(targets))
    features=None
    
    for eps in range(start_ep, args.epochs):
        #Pseudo loss filtering on the noisy examples
        if eps > args.warmup and eps != start_ep and not args.no_weights and preds.sum() > 2:
            _trainer.weights = torch.ones(len(_trainer.weights))
            preds = _trainer.is_idn #Detected noisy samples
            
            interest = min_max(pseudo_loss[preds]) #Pseudo loss of noisy samples
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm = gmm.fit(interest.reshape(-1,1))
            #Compute confidence in the pseudo-label
            proba_ = gmm.predict_proba(interest.reshape(-1,1))[:, 0]
            if gmm.means_[1] < gmm.means_[0]:
                proba_ = 1-proba_
            w = torch.from_numpy(proba_).float() #w=1 means the pseudo-label can be 100% trusted                
               
            _trainer.weights[preds] = w            
            
        #Training
        if (args.resume is not None and eps > start_ep) or args.resume is None: #If training was resumed, compute the losses first
            _trainer.train(eps)                       
                
        if eps >= args.warmup-1:
            features, losses, pseudo_loss = _trainer.track_loss()
            
            #Filtering of noisy samples using a simple GMM on the training loss
            interest = min_max(losses)
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm = gmm.fit(interest.reshape(-1,1))                   
            proba = gmm.predict_proba(interest.reshape(-1,1))[:, 0] #Probability to belong to the high loss mode (noisy)
            if gmm.means_[0] < gmm.means_[1]:
                proba = 1-proba                        
                        
            preds = torch.from_numpy(proba > args.thresh).bool() #Fixed thresold on the noisiness            
            
            _trainer.is_clean = ~preds
            _trainer.is_idn = preds                
                
        acc_c = _trainer.val(eps, f=features) #f=None will skip the kNN evaluation
            
if __name__ == "__main__":
   main()

   
