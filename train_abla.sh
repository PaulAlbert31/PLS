mkdir logs

#40% id CIFAR-100
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 200 > logs/logs_40id_abla_mixup.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 --no-reg --no-weights > logs/logs_40id_abla_correct.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 --no-weights > logs/logs_40id_abla_correctreg.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 > logs/logs_40id_abla_correctregweights.txt
#PLS
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_40id_abla_correctregweightscont.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 --proj-size 128 --cont --no-weights > logs/logs_40id_abla_noweights.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 200 --proj-size 128 --cont --no-correct > logs/logs_40id_abla_nocorrect.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 --proj-size 128 --cont --no-weights-cont > logs/logs_40id_abla_noweightscont.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_40id --ood-noise 0.0 --id-noise 0.4 --warmup 30 --proj-size 128 --cont > logs/logs_40id_abla_nomixup.txt


#20o20i CIFAR-100
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 200 > logs/logs_20o20id_abla_mixup.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --no-reg --no-weights > logs/logs_20o20id_abla_correct.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --no-weights > logs/logs_20o20id_abla_correctreg.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 > logs/logs_20o20id_abla_correctregweights.txt
#PLS
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont > logs/logs_20o20id_abla_correctregweightscont.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont --no-weights > logs/logs_20o20id_abla_noweights.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 200 --proj-size 128 --cont --no-correct > logs/logs_20o20id_abla_nocorrect.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --mixup --warmup 30 --proj-size 128 --cont --no-weights-cont > logs/logs_20o20id_abla_noweightscont.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed 1 --exp-name cifar100_20o20id --ood-noise 0.2 --id-noise 0.2 --warmup 30 --proj-size 128 --cont > logs/logs_20o20id_abla_nomixup.txt
