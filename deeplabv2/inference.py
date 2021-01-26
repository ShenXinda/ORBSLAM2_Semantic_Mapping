#!/usr/bin/env python
# -*- coding: utf8 *-*

from __future__ import absolute_import, division, print_function

import os
import cv2
# import matplotlib
# matplotlib.use('TkAgg') # 加上这句，避免“g_main_context_push_thread_default: assertion 'acquired_context' failed”错误
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from libs.models import *
from libs.utils import DenseCRF

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize （推断时不进行resize）
    # scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    # image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )
    
    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)
    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    # print(image.shape)
    logits = model(image)
    # print(logits.shape)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()
    
    # np.save("probmap.npy",probs)
    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)   # crf.py

    labelmap = np.argmax(probs, axis=0)

    return probs, labelmap

class SegModel():
    def __init__(self):
        config_path = "deeplabv2/configs/cocostuff164k.yaml"
        model_path = "deeplabv2/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
        # config_path = "/home/xshen/my_workspace/deeplab-pytorch/configs/cocostuff164k.yaml"
        # model_path = "/home/xshen/my_workspace/deeplab-pytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
        # config_path = "/home/xshen/my_workspace/deeplab-pytorch/configs/cocostuff10k.yaml"
        # model_path = "/home/xshen/my_workspace/deeplab-pytorch/checkpoint_final_10k_uotstride8.pth"
        cuda = True
        crf = False  # 使用crf会有问题？？

        print(os.getcwd())
        self.CONFIG = OmegaConf.load(config_path) 
        self.device = get_device(cuda)
        torch.set_grad_enabled(False)
        classes = get_classtable(self.CONFIG)
        self.postprocessor = setup_postprocessor(self.CONFIG) if crf else None
        
        self.model = eval(self.CONFIG.MODEL.NAME)(n_classes=self.CONFIG.DATASET.N_CLASSES)
        # print(self.model) # 显示模型
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage) 

        # keys = []
        # for key in state_dict.keys():
        #     keys.append(key)

        # for key in keys:
        #     if key[5:12] != 'module1' or key[5:12] != 'module2' :
        #         if key[5:9] =='aspp':
        #             state_dict[key[0:5]+'module2'+key[4:]] = state_dict.pop(key)
        #         else:
        #             state_dict[key[0:5]+'module1'+key[4:]] = state_dict.pop(key)
                
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        print("Model:", self.CONFIG.MODEL.NAME)

    def runModel(self, image):
        image, raw_image = preprocessing(image, self.device, self.CONFIG)  
        probs, labelmap = inference(self.model, image, raw_image, self.postprocessor)
        # print(probs.shape)
        # print(np.max(probs,0))
        prob = [probs]
        return prob

    def getMap(self, image):
        image, raw_image = preprocessing(image, self.device, self.CONFIG)  
        probs, labelmap = inference(self.model, image, raw_image, self.postprocessor)
        mp = [labelmap]
        return mp


def init():
    global seg_model 
    seg_model =  SegModel()

def runSegModelTest(image):
    return seg_model.runModel(image)
    
def runSegModel(array):
    img = array[:, 0:len(array[0] - 2):3]
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return seg_model.runModel(image)

def getLabelMap(array):
    img = array[:, 0:len(array[0] - 2):3]
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return seg_model.getMap(image)

def run():
    """
    Inference from a single image
    """
    config_path = "deeplabv2/configs/cocostuff164k.yaml"
    model_path = "deeplabv2/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
    image_path = "deeplabv2/test"
    cuda = True
    crf = False  # # 使用crf会有问题？？


    # Setup
    CONFIG = OmegaConf.load(config_path)  # 解析yaml文件
    device = get_device(cuda)
    torch.set_grad_enabled(False)
    classes = get_classtable(CONFIG) # 类标签，保存成dict形式
    postprocessor = setup_postprocessor(CONFIG) if crf else None  # 是否进行CRF后处理

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES) # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage) 
    model.load_state_dict(state_dict) # 模型加载参数
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Inference
    # 思路：判断image是否为一个路径，若是，则推断路径下所有图片
    if os.path.isdir(image_path):

        for filename in os.listdir(image_path):
            filename = image_path+filename if str(image_path).endswith('/') else image_path+'/'+filename
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            image, raw_image = preprocessing(image, device, CONFIG)  
            # probs是概率图，labelmap是类别图
            # print(image)
            probs, labelmap = inference(model, image, raw_image, postprocessor)
            print(probs)

    else:
        print("请输入一个存放图像的路径！")


if __name__ == "__main__":
    # run()
    image_path = "deeplabv2/test"
    # seg_model = SegModel()
    init()
    if os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            filename = image_path+filename if str(image_path).endswith('/') else image_path+'/'+filename
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            # probmap = seg_model.runModel(image)
            probmap  = runSegModelTest(image)
            print(probmap)
