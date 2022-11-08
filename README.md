# Pseudo-Label Selection for label noise (PLS)
Official repository for Is your noise correction noisy? PLS: Robustness to label noise with two stage detection WACV 2023 [paper](https://arxiv.org/abs/2210.04578)

[!PLS][PLS.png]

### Dependencies

    conda env create -f env.yml
    conda activate pls
pytorch=1.7.1, torchvision=8.2, cuda=10.2, python=3.8


### Dataset setup
Set the path to your datasets in the `mypath.py` file
Download the web fine-grained from [here](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset)

## How to use
Run PLS on CIFAR-100 with 40% of ID noise

    python main.py --dataset cifar100 --epochs 200 --batch-size 256 --net preresnet18 --lr 0.1 --exp-name cifar100_40idnoise --ood-noise 0.0 --id-noise 0.4 --mixup --warmup 30 --cont

The various  `train*.sh`  files list multiple example commands to run PLS on CIFAR-100, miniImageNet corrupted with web noise and the web fine-grained datasets.

## Train on a custom dataset
Edit the  `datasets/custom.py` (dataset creation), the `utils.py` (mean, std and image size) and the `mypath.py` (dataset path) files to fit your custom dataset and specify the `--dataset custom` command when running the code.

## Some results from the paper
Controlled Noisy Web Labels (CNWL) dataset
|r_out| 0.2 | 0.4 | 0.6 | 0.8 |
| ------ | ------ | ------ | ------ | ------ |
|top-1 acc|63.10| 60.02 | 54.41 | 46.51 |
|std 3 runs| 0.14 |0.15| 0.49| 0.20|

CIFAR-100 ID noise
|r_in| 0.0 | 0.2 | 0.5 | 0.8 |
| ------ | ------ | ------ | ------ | ------ |
|top-1 acc|78.85|80.03|76.48|63.33|
|std 3 runs|0.21|0.15|0.25|0.38|

CIFAR-100 ID and OOD noise (ImageNet32)
|r_in| 0.2 | 0.2 | 0.2 | 0.4 |
| ------ | ------ | ------ | ------ | ------ |
|r_out| 0.2 | 0.4 | 0.6 | 0.4 |
|top-1 acc|76.29|72.06|57.78|56.92|
|std 3 runs|0.28|0.19|0.26|0.49|

Web-fg datasets
|dataset| web-aircraft | web-bird | web-car |
| ------ | ------ | ------ | ------ |
|top-1 acc|87.58|79.00|86.27|


## Cite our paper if it helps your research
```
@inproceedings{2023_WACV_PLS,
  title="{Is your noise correction noisy? PLS: Robustness to label noise with two stage detection}",
  author="Albert, Paul and Arazo, Eric and Kirshna, Tarun and O'Connor, Noel E and McGuinness, Kevin",
  booktitle="{Winter Conference on Applications of Computer Vision (WACV)}",
  year={2023}
}
```
