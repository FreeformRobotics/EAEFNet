# EAEFNet

![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.10.0](https://img.shields.io/badge/PyTorch-1.10.0-blue)


## Introduction

![a1018965db81629d43805608d569c0a](https://user-images.githubusercontent.com/45811724/208004358-395e5078-6f63-43f9-b6f5-426736c69b1a.png)

EFEANet
This is the official pytorch implementation of EAEFNet: Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks. Some of the codes are borrowed from [MFNet](https://github.com/haqishen/MFNet-pytorch) , [RTFNet](https://github.com/yuxiangsun/RTFNet) and [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting). The master branch works with **PyTorch 1.10+**.

## Installation

**Crow-Counting**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_RGBT-CC
pip install -r requirements.txt
```

**Detection**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
pip install openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc3,<3.1.0"
cd EAEFNet_Detection/EAEF_mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO , don't forget it!
mim install -v -e .
```

**Segmentation**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_RGBT_MF
pip install -r requirements.txt
```

## Data prepare
![1671154999406](https://user-images.githubusercontent.com/45811724/208002737-71390486-a4c7-4f6f-b225-c259acb4e41c.png)


MFTNet Dataset: http://gofile.me/4jm56/CfukComo1

PST900 Dataset: https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view

M3FD Dataset: https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6

RGBT-CC Dataset: https://www.dropbox.com/sh/o4ww2f5tv3nay9n/AAA4CfVMTZcdwsFxFlhwDsSba?dl=0

###### Ps: You also can download them from [here](https://drive.google.com/drive/folders/1fqNwaumH0BrcAIvS0ebAjS35LX31Yw4S?usp=share_link)!

## Train-model Download 


| Task         | Dataset | model   | mIoU   | Train Download                                               |
| ------------ | ------- | ------- | ------ | ------------------------------------------------------------ |
| Segmentation | MFNet   | EAEFNet | 58.91% | https://drive.google.com/drive/folders/12ONwVaaO35VbW7rZ83P-pSVWp_bFiPhv?usp=share_link |
| Segmentation | PSP900  | EAEFNet | 85.56% | https://drive.google.com/drive/folders/1Czm7vtmaW6fTCk4fBAfO2OAWoHrJry9Z?usp=share_link |

| Task      | Dataset | model       | mAP@0.5 | Train Download                                               |
| --------- | ------- | ----------- | ------- | ------------------------------------------------------------ |
| Detection | M3FD    | EAEF+Yolov5 | 80.4%   | https://drive.google.com/drive/folders/1JcvZUmTUB936H9JoYjYrM9H-jHKnjNzc?usp=share_link |

| Task          | Dataset | model | RMSE   | Train Download                                               |
| ------------- | ------- | ----- | ------ | ------------------------------------------------------------ |
| Crow-counting | RGBT-CC | EAEF  | 26.99% | https://drive.google.com/drive/folders/1eb0GwISb0AUULrDpUo8jBZC5Oh4zShgD?usp=share_link |


## Test
#### M3FD Detection
```
python EAEF_mmyolo/tools/test.py yolov5/bi_yolov5 
```
#### MFNet Segmentation
```
python EAEF_MF/run_own_pth.py
```
#### PST900 Segmentation
```
python EAEF_PST/run_own_pth.py
```
#### RGBT-CC Crow-counting
```
python EAEF_CC/test.py
```



