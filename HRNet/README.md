# HRNet
해당 모델은 Keypoint 예측시 활용한 모델입니다.

## Installation
pytorch: 1.9.0+cu102
albumentations: 0.4.6
```
pip install albumentations==0.4.6
```
## Model
학습시 [leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)에서 제공한 pretrained 모델 중 `pose_hrnet_w48_384x288.pth`를 사용하였습니다.

## Structure
```
HRNet
│  README.md
│  requirements.txt
│  config.py
│  data.py
│  eval.py
│  inference.py
│  loss.py
│  model.py
│  seed.py
│  train.py
│  transform.py
│  __init__.py
│
├─config
│      heatmap_train.yaml
│
├─pretrain
│      pose_hrnet_w48_384x288.pth
│      
└─weight
```