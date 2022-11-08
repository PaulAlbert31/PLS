#You will need more than one GPU for these
python main.py --net resnet50 --dataset web-bird --epochs 110 --batch-size 16 --lr 0.003 --seed 1 --exp-name web-bird --warmup 10 --proj-size 128 --cont --mixup > logs/logs_web_bird.sh
python main.py --net resnet50 --dataset web-car --epochs 110 --batch-size 16 --lr 0.003 --seed 1 --exp-name web-car --warmup 10 --proj-size 128 --cont --mixup > logs/logs_web_car.sh
python main.py --net resnet50 --dataset web-aircraft --epochs 110 --batch-size 16 --lr 0.003 --seed 1 --exp-name web-aircraft --warmup 10 --proj-size 128 --cont --mixup > logs/logs_web_aircraft.sh
