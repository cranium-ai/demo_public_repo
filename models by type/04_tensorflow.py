"""
04_tensorflow.py
================
TensorFlow / Keras model calls — alphabetical.
These are the TF-native equivalents of the vision models listed without '/'.

Install:
    pip install tensorflow tensorflow-hub
"""

import tensorflow as tf
import numpy as np

_img_224 = tf.random.normal([1, 224, 224, 3])
_img_299 = tf.random.normal([1, 299, 299, 3])

# inception_resnet_v2
tf_irv2 = tf.keras.applications.InceptionResNetV2(weights=None)
tf_irv2.predict(_img_299)

# inception_v3
tf_iv3 = tf.keras.applications.InceptionV3(weights=None)
tf_iv3.predict(_img_299)

# mobilenet_v1
tf_mnv1 = tf.keras.applications.MobileNet(weights=None)
tf_mnv1.predict(_img_224)

# mobilenet_v1_025
tf_mnv1_025 = tf.keras.applications.MobileNet(weights=None, alpha=0.25)
tf_mnv1_025.predict(_img_224)

# mobilenet_v1_050
tf_mnv1_050 = tf.keras.applications.MobileNet(weights=None, alpha=0.5)
tf_mnv1_050.predict(_img_224)

# mobilenet_v2
tf_mnv2 = tf.keras.applications.MobileNetV2(weights=None)
tf_mnv2.predict(_img_224)

# resnet_v1_50  (Keras ResNet50 = v1)
tf_rn50 = tf.keras.applications.ResNet50(weights=None)
tf_rn50.predict(_img_224)

# resnet_v1_101
tf_rn101 = tf.keras.applications.ResNet101(weights=None)
tf_rn101.predict(_img_224)

# resnet_v1_152
tf_rn152 = tf.keras.applications.ResNet152(weights=None)
tf_rn152.predict(_img_224)

# resnet_v2_50
tf_rn50v2 = tf.keras.applications.ResNet50V2(weights=None)
tf_rn50v2.predict(_img_224)

# resnet_v2_101
tf_rn101v2 = tf.keras.applications.ResNet101V2(weights=None)
tf_rn101v2.predict(_img_224)

# resnet_v2_152
tf_rn152v2 = tf.keras.applications.ResNet152V2(weights=None)
tf_rn152v2.predict(_img_224)

# vgg_16
tf_vgg16 = tf.keras.applications.VGG16(weights=None)
tf_vgg16.predict(_img_224)

# vgg_19
tf_vgg19 = tf.keras.applications.VGG19(weights=None)
tf_vgg19.predict(_img_224)
