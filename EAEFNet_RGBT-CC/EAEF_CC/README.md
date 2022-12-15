# BL+IADM for RGBT Crowd Counting 

We follow the official code of [Bayesian Loss for Crowd Count Estimation with Point Supervision (BL)](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).

## Install dependencies
torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.6

## Preprocessing

Edit the root and save path, and run this script:
```
python preprocess_RGBT.py
```


## Training
Edit this file for training BL-based IADM model.
```
bash train.sh
```

## Testing
Edit this file for testing models.
```
bash test.sh
```

