"""
03_torchvision_timm.py
======================
Vision models without a '/' in their name — torchvision, timm, detectron2,
and ultralytics (YOLO).  All calls are alphabetical.

Install:
    pip install torch torchvision timm ultralytics Pillow
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
"""

import torch
import torchvision.models as tv_models
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
import timm

_t224 = torch.randn(1, 3, 224, 224)
_t299 = torch.randn(1, 3, 299, 299)
_t300 = torch.randn(1, 3, 300, 300)
_img  = _t224.squeeze(0)          # single-image tensor for detection models

DUMMY_IMAGE_PATH = "sample.jpg"   # provide a real image for YOLO at runtime

# alexnet
alexnet_model = tv_models.alexnet(weights=None)
alexnet_model.eval()
alexnet_model(_t224)

# cascade_mask_rcnn_R_50_FPN_1x  — detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg_d2 = get_cfg()
cfg_d2.merge_from_file(
    model_zoo.get_config_file("cascade_mask_rcnn_R_50_FPN_1x")
)
cfg_d2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml"
)
cfg_d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_d2.MODEL.DEVICE = "cpu"
predictor_cascade = DefaultPredictor(cfg_d2)
# predictor_cascade(cv2_bgr_image)  # pass a BGR numpy array at runtime

# fasterrcnn_resnet50_fpn
fasterrcnn_model = fasterrcnn_resnet50_fpn(weights=None)
fasterrcnn_model.eval()
fasterrcnn_model([_img])

# fcos_resnet50_fpn
fcos_model = fcos_resnet50_fpn(weights=None)
fcos_model.eval()
fcos_model([_img])

# inception_resnet_v2  — timm
inception_resnet_v2_model = timm.create_model("inception_resnet_v2", pretrained=False)
inception_resnet_v2_model.eval()
inception_resnet_v2_model(_t299)

# inception_v3  — torchvision
from torchvision.models import inception_v3
inception_v3_model = inception_v3(weights=None)
inception_v3_model.eval()
inception_v3_model(_t299)

# maskrcnn_resnet50_fpn
maskrcnn_model = maskrcnn_resnet50_fpn(weights=None)
maskrcnn_model.eval()
maskrcnn_model([_img])

# mobilenet_v1  — timm (not in torchvision)
mobilenet_v1_model = timm.create_model("mobilenetv1_100", pretrained=False)
mobilenet_v1_model.eval()
mobilenet_v1_model(_t224)

# mobilenet_v1_025
mobilenet_v1_025_model = timm.create_model("mobilenetv1_025", pretrained=False)
mobilenet_v1_025_model.eval()
mobilenet_v1_025_model(_t224)

# mobilenet_v1_050
mobilenet_v1_050_model = timm.create_model("mobilenetv1_050", pretrained=False)
mobilenet_v1_050_model.eval()
mobilenet_v1_050_model(_t224)

# mobilenet_v2  — torchvision
from torchvision.models import mobilenet_v2
mobilenet_v2_model = mobilenet_v2(weights=None)
mobilenet_v2_model.eval()
mobilenet_v2_model(_t224)

# mobilenet
from torchvision.models import mobilenet
mobilenet_model = mobilenet(weights=None)
mobilenet_model.eval()
mobilenet_model(_t224)

# resnet_v1_50  — torchvision ResNet50 (v1 by default)
resnet_v1_50_model = tv_models.resnet50(weights=None)
resnet_v1_50_model.eval()
resnet_v1_50_model(_t224)

# resnet_v1_101
resnet_v1_101_model = tv_models.resnet101(weights=None)
resnet_v1_101_model.eval()
resnet_v1_101_model(_t224)

# resnet_v1_152
resnet_v1_152_model = tv_models.resnet152(weights=None)
resnet_v1_152_model.eval()
resnet_v1_152_model(_t224)

# resnet_v2_50  — timm pre-activation ResNet
resnet_v2_50_model = timm.create_model("resnet_v2_50", pretrained=False)
resnet_v2_50_model.eval()
resnet_v2_50_model(_t224)

# resnet_v2_101
resnet_v2_101_model = timm.create_model("resnet_v2_101", pretrained=False)
resnet_v2_101_model.eval()
resnet_v2_101_model(_t224)

# resnet_v2_152
resnet_v2_152_model = timm.create_model("resnet_v2_152", pretrained=False)
resnet_v2_152_model.eval()
resnet_v2_152_model(_t224)

# retinanet_resnet50_fpn
retinanet_model = retinanet_resnet50_fpn(weights=None)
retinanet_model.eval()
retinanet_model([_img])

# ssd_inception_v2_coco  — timm (closest public PyTorch equivalent)
ssd_inception_v2_model = timm.create_model("ssd_inception_v2_coco", pretrained=False)
ssd_inception_v2_model.eval()
ssd_inception_v2_model(_t224)

# ssd300_vgg16
ssd300_model = ssd300_vgg16(weights=None)
ssd300_model.eval()
ssd300_model([_t300.squeeze(0)])

# vgg_16
vgg16_model = tv_models.vgg16(weights=None)
vgg16_model.eval()
vgg16_model(_t224)

# vgg_19
vgg19_model = tv_models.vgg19(weights=None)
vgg19_model.eval()
vgg19_model(_t224)

# yolov10b  — ultralytics
from ultralytics import YOLO
yolov10b_model = YOLO("yolov10b.pt")
yolov10b_model.predict(DUMMY_IMAGE_PATH, verbose=False)

# yolov3  — ultralytics
yolov3_model = YOLO("yolov3.pt")
yolov3_model.predict(DUMMY_IMAGE_PATH, verbose=False)
