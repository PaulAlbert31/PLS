mkdir logs
for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name miniimagenet_20 --noise-ratio 0.2 --mixup --warmup 1 --proj-size 128 --cont > logs/logs_mini_20_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name miniimagenet_40 --noise-ratio 0.4 --mixup --warmup 1 --proj-size 128 --cont > logs/logs_mini_40_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name miniimagenet_60 --noise-ratio 0.6 --mixup --warmup 1 --proj-size 128 --cont > logs/logs_mini_60_seed$SEED.txt
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --seed $SEED --exp-name miniimagenet_80 --noise-ratio 0.8 --mixup --warmup 1 --proj-size 128 --cont > logs/logs_mini_80_seed$SEED.txt
done
