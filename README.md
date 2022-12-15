# EAEFNet

![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.11.0](https://img.shields.io/badge/PyTorch-1.11.0-blue)


## Introduction

The master branch works with **PyTorch 1.8+**.

## Installation

**Crow-Counting**

EAEFNet achieves RGBT-CC tasks based on the  RGBT-CC benchmark.

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

MFTNet Dataset: http://gofile.me/4jm56/CfukComo1

PST900 Dataset: https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view

M3FD Dataset: https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6

RGBT-CC Dataset: https://www.dropbox.com/sh/o4ww2f5tv3nay9n/AAA4CfVMTZcdwsFxFlhwDsSba?dl=0

###### Ps: You also can download it from [here](https://drive.google.com/drive/folders/1fqNwaumH0BrcAIvS0ebAjS35LX31Yw4S?usp=share_link)!

## Pre-model Download 

MFNet : https://drive.google.com/drive/folders/12ONwVaaO35VbW7rZ83P-pSVWp_bFiPhv?usp=share_link

PST900 : https://drive.google.com/drive/folders/1Czm7vtmaW6fTCk4fBAfO2OAWoHrJry9Z?usp=share_link

M3FD : https://drive.google.com/drive/folders/1JcvZUmTUB936H9JoYjYrM9H-jHKnjNzc?usp=share_link

RGBT-CC: https://drive.google.com/drive/folders/1eb0GwISb0AUULrDpUo8jBZC5Oh4zShgD?usp=share_link


## Test

```
python EAEF_mmyolo/tools/test.py yolov5/bi_yolov5 
```

```
python EAEF_MF/run_own_pth.py
```

```
python EAEF_PST/run_own_pth.py
```

```
python EAEF_CC/test.py
```

## Update
