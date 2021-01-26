# ORBSLAM2_Semantic_Mapping
Semantic mapping based on pixel level classification.

## 1. Reference

- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
- [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

## 2. Prerequisites

The weight file should be placed in `deeplabv2/data/models`，and downloads in [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) . And you can change the path in `deeplabv2/inference.py`.

```python
class SegModel():
    def __init__(self):
        config_path = "deeplabv2/configs/cocostuff164k.yaml"
        model_path = "deeplabv2/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
        ......
```

## 3. Build and Run

The same as [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

## 4. Explain (Chinese)

Refer to [基于ORB-SLAM2的语义地图构建](https://blog.csdn.net/XindaBlack/article/details/113097652).

