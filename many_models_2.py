# =============================================================================
# 4.  TORCHVISION / TIMM MODELS  (best-judgement vision models without '/')
# =============================================================================

import torch
import torchvision.models as tv_models
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
from PIL import Image
import torchvision.transforms as T

_dummy_tensor  = torch.randn(1, 3, 224, 224)
_dummy_tensor_300 = torch.randn(1, 3, 300, 300)

# alexnet
alexnet_model = tv_models.alexnet(weights=None)
alexnet_model.eval()
alexnet_model(_dummy_tensor)

# cascade_mask_rcnn_R_50_FPN_1x  – available via detectron2
# (torchvision does not ship this; use detectron2)
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg_d2 = get_cfg()
cfg_d2.merge_from_file(
    model_zoo.get_config_file(
        "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml"
    )
)
cfg_d2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml"
)
cfg_d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_d2.MODEL.DEVICE = "cpu"
predictor_cascade = DefaultPredictor(cfg_d2)
# predictor_cascade(cv2_image)   # pass a BGR numpy array at runtime

# fasterrcnn_resnet50_fpn
fasterrcnn_model = fasterrcnn_resnet50_fpn(weights=None)
fasterrcnn_model.eval()
fasterrcnn_model([_dummy_tensor.squeeze(0)])

# fcos_resnet50_fpn
fcos_model = fcos_resnet50_fpn(weights=None)
fcos_model.eval()
fcos_model([_dummy_tensor.squeeze(0)])

# inception_resnet_v2  – available via timm
import timm
inception_resnet_v2_model = timm.create_model("inception_resnet_v2", pretrained=False)
inception_resnet_v2_model.eval()
inception_resnet_v2_model(torch.randn(1, 3, 299, 299))

# inception_v3
inception_v3_model = tv_models.inception_v3(weights=None)
inception_v3_model.eval()
inception_v3_model(torch.randn(1, 3, 299, 299))

# maskrcnn_resnet50_fpn
maskrcnn_model = maskrcnn_resnet50_fpn(weights=None)
maskrcnn_model.eval()
maskrcnn_model([_dummy_tensor.squeeze(0)])

# mobilenet_v1  – not in torchvision; use timm
mobilenet_v1_model = timm.create_model("mobilenetv1_100", pretrained=False)
mobilenet_v1_model.eval()
mobilenet_v1_model(_dummy_tensor)

# mobilenet_v1_025
mobilenet_v1_025_model = timm.create_model("mobilenetv1_025", pretrained=False)
mobilenet_v1_025_model.eval()
mobilenet_v1_025_model(_dummy_tensor)

# mobilenet_v1_050
mobilenet_v1_050_model = timm.create_model("mobilenetv1_050", pretrained=False)
mobilenet_v1_050_model.eval()
mobilenet_v1_050_model(_dummy_tensor)

# mobilenet_v2
mobilenet_v2_model = tv_models.mobilenet_v2(weights=None)
mobilenet_v2_model.eval()
mobilenet_v2_model(_dummy_tensor)

# resnet_v1_50  – use torchvision resnet50 (ResNet-v1 by default)
resnet_v1_50_model = tv_models.resnet_v1_50(weights=None)
resnet_v1_50_model.eval()
resnet_v1_50_model(_dummy_tensor)

# resnet_v1_101
resnet_v1_101_model = tv_models.resnet_v1_101(weights=None)
resnet_v1_101_model.eval()
resnet_v1_101_model(_dummy_tensor)

# resnet_v1_152
resnet_v1_152_model = tv_models.resnet_v1_152(weights=None)
resnet_v1_152_model.eval()
resnet_v1_152_model(_dummy_tensor)

# resnet_v2_50  – use timm (pre-activation ResNet)
resnet_v2_50_model = timm.create_model("resnetv2_50", pretrained=False)
resnet_v2_50_model.eval()
resnet_v2_50_model(_dummy_tensor)

# resnet_v2_101
resnet_v2_101_model = timm.create_model("resnetv2_101", pretrained=False)
resnet_v2_101_model.eval()
resnet_v2_101_model(_dummy_tensor)

# resnet_v2_152
resnet_v2_152_model = timm.create_model("resnetv2_152", pretrained=False)
resnet_v2_152_model.eval()
resnet_v2_152_model(_dummy_tensor)

