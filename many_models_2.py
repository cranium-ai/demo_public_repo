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
resnet_v1_50_model = tv_models.resnet50(weights=None)
resnet_v1_50_model.eval()
resnet_v1_50_model(_dummy_tensor)

# resnet_v1_101
resnet_v1_101_model = tv_models.resnet101(weights=None)
resnet_v1_101_model.eval()
resnet_v1_101_model(_dummy_tensor)

# resnet_v1_152
resnet_v1_152_model = tv_models.resnet152(weights=None)
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

# retinanet_resnet50_fpn
retinanet_model = retinanet_resnet50_fpn(weights=None)
retinanet_model.eval()
retinanet_model([_dummy_tensor.squeeze(0)])

# ssd_inception_v2_coco  – use timm (closest public equivalent)
ssd_inception_v2_model = timm.create_model("ssd_inception_v2", pretrained=False)
ssd_inception_v2_model.eval()
ssd_inception_v2_model(_dummy_tensor)

# ssd300_vgg16
ssd300_model = ssd300_vgg16(weights=None)
ssd300_model.eval()
ssd300_model([_dummy_tensor_300.squeeze(0)])

# vgg_16
vgg16_model = tv_models.vgg16(weights=None)
vgg16_model.eval()
vgg16_model(_dummy_tensor)

# vgg_19
vgg19_model = tv_models.vgg19(weights=None)
vgg19_model.eval()
vgg19_model(_dummy_tensor)

# yolov10b  – ultralytics
from ultralytics import YOLO
yolov10b_model = YOLO("yolov10b.pt")
yolov10b_model.predict(DUMMY_IMAGE_PATH, verbose=False)

# yolov3  – ultralytics
yolov3_model = YOLO("yolov3.pt")
yolov3_model.predict(DUMMY_IMAGE_PATH, verbose=False)


# =============================================================================
# 5.  TENSORFLOW / KERAS MODELS  (best-judgement TF model zoo)
# =============================================================================

import tensorflow as tf

# inception_resnet_v2  (TF/Keras built-in)
# Note: also modelled via timm above; this shows the TF route
tf_irv2 = tf.keras.applications.InceptionResNetV2(weights=None)
tf_irv2.predict(tf.random.normal([1, 299, 299, 3]))

# inception_v3  (TF/Keras built-in)
tf_iv3 = tf.keras.applications.InceptionV3(weights=None)
tf_iv3.predict(tf.random.normal([1, 299, 299, 3]))

# mobilenet_v1  (TF/Keras built-in)
tf_mnv1 = tf.keras.applications.MobileNet(weights=None)
tf_mnv1.predict(tf.random.normal([1, 224, 224, 3]))

# mobilenet_v1_025
tf_mnv1_025 = tf.keras.applications.MobileNet(weights=None, alpha=0.25)
tf_mnv1_025.predict(tf.random.normal([1, 224, 224, 3]))

# mobilenet_v1_050
tf_mnv1_050 = tf.keras.applications.MobileNet(weights=None, alpha=0.5)
tf_mnv1_050.predict(tf.random.normal([1, 224, 224, 3]))

# mobilenet_v2  (TF/Keras built-in)
tf_mnv2 = tf.keras.applications.MobileNetV2(weights=None)
tf_mnv2.predict(tf.random.normal([1, 224, 224, 3]))

# resnet_v1_50  (TF/Keras ResNet50 = v1)
tf_rn50 = tf.keras.applications.ResNet50(weights=None)
tf_rn50.predict(tf.random.normal([1, 224, 224, 3]))

# resnet_v1_101
tf_rn101 = tf.keras.applications.ResNet101(weights=None)
tf_rn101.predict(tf.random.normal([1, 224, 224, 3]))

# resnet_v1_152
tf_rn152 = tf.keras.applications.ResNet152(weights=None)
tf_rn152.predict(tf.random.normal([1, 224, 224, 3]))

# resnet_v2_50
tf_rn50v2 = tf.keras.applications.ResNet50V2(weights=None)
tf_rn50v2.predict(tf.random.normal([1, 224, 224, 3]))

# resnet_v2_101
tf_rn101v2 = tf.keras.applications.ResNet101V2(weights=None)
tf_rn101v2.predict(tf.random.normal([1, 224, 224, 3]))

# resnet_v2_152
tf_rn152v2 = tf.keras.applications.ResNet152V2(weights=None)
tf_rn152v2.predict(tf.random.normal([1, 224, 224, 3]))

# ssd_inception_v2_coco  – TF Object Detection API (tf1 hub model)
import tensorflow_hub as hub
ssd_inception_v2_tf = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"  # closest public TFHub equivalent
)

# vgg_16
tf_vgg16 = tf.keras.applications.VGG16(weights=None)
tf_vgg16.predict(tf.random.normal([1, 224, 224, 3]))

# vgg_19
tf_vgg19 = tf.keras.applications.VGG19(weights=None)
tf_vgg19.predict(tf.random.normal([1, 224, 224, 3]))