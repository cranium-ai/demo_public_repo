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
yolov10b_model = YOLO("yolov10b")
yolov10b_model.predict(DUMMY_IMAGE_PATH, verbose=False)

# yolov3  – ultralytics
yolov3_model = YOLO("yolov3")
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
tf_vgg16 = tf.keras.applications.vgg16(weights=None)
tf_vgg16.predict(tf.random.normal([1, 224, 224, 3]))

# vgg_19
tf_vgg19 = tf.keras.applications.vgg19(weights=None)
tf_vgg19.predict(tf.random.normal([1, 224, 224, 3]))