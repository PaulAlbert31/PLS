mkdir logs
for SEED in 1 2 3
do
    #ID noise only
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.0 --id-noise 0.0 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_0id_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.0 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_20id_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.0 --id-noise 0.5 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_50id_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.0 --id-noise 0.8 --mixup --warmup 30 --proj-size 128 --cont --thresh .5 > logs/logs_80id_seed$SEED.txt

    #ID and OOD from ImageNet
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_20id_20ood_inet_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.4 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_20id_40ood_inet_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.6 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_20id_60ood_inet_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.4 --id-noise 0.4 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_40id_40ood_inet_seed$SEED.txt
    
    #ID and OOD from Places365
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont --corruption places > logs/logs_20id_20ood_places_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.4 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont --corruption places > logs/logs_20id_40ood_places_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.6 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont --corruption places > logs/logs_20id_60ood_places_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name cifar100_seeds --ood-noise 0.4 --id-noise 0.4 --mixup --warmup 30 --proj-size 128 --cont --corruption places > logs/logs_40id_40ood_places_seed$SEED.txt
    
done

