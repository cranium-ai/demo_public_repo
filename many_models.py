"""
model_calls.py
==============
Demonstrates how to call each model using the appropriate library.
Models are organized alphabetically within each section, and the
sections themselves appear in the order that their first model
occurs alphabetically across all models.

Library routing rules
---------------------
- Contains  '/'  (namespace/repo)    → Hugging Face (transformers / pipeline)
- Starts with 'gemini'               → Google Generative AI (google-generativeai)
- Starts with 'claude'               → Anthropic SDK
- Starts with 'gpt' or 'o<int>-'    → OpenAI SDK
- sklearn model names                → scikit-learn
- Everything else                    → best-judgement
                                       (torchvision, timm, tensorflow/keras,
                                        ultralytics, catboost, lightgbm, xgboost …)

NOTE: API keys are read from environment variables; no credentials are
      hard-coded.  Install dependencies before running:

  pip install anthropic openai google-generativeai transformers torch \
              torchvision timm tensorflow scikit-learn catboost lightgbm \
              xgboost ultralytics Pillow
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard-library / environment helpers
# ─────────────────────────────────────────────────────────────────────────────
import os

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY",  "YOUR_ANTHROPIC_API_KEY")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY",     "YOUR_GOOGLE_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY",     "YOUR_OPENAI_API_KEY")

DUMMY_TEXT         = "Hello, world!"
DUMMY_PROMPT       = "What is the capital of France?"
DUMMY_IMAGE_PATH   = "sample.jpg"          # provide a real image when running


# =============================================================================
# 0.  SKLEARN MODELS  (alphabetical)
# =============================================================================

from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    Ridge,
    SGDRegressor,
)
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.naive_bayes import GaussianNB
import numpy as np

_X_train = np.random.rand(20, 4)
_y_cls   = np.random.randint(0, 2, 20)
_y_reg   = np.random.rand(20)

# BayesianRidge
bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(_X_train, _y_reg)
bayesian_ridge.predict(_X_train[:1])

# DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(_X_train, _y_cls)
decision_tree_classifier.predict(_X_train[:1])

# DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(_X_train, _y_reg)
decision_tree_regressor.predict(_X_train[:1])

# ElasticNet
elastic_net = ElasticNet()
elastic_net.fit(_X_train, _y_reg)
elastic_net.predict(_X_train[:1])

# Extra Tree Classifier  (sklearn calls this ExtraTreesClassifier)
extra_tree_classifier = ExtraTreesClassifier()
extra_tree_classifier.fit(_X_train, _y_cls)
extra_tree_classifier.predict(_X_train[:1])

# Extra Tree Regressor
extra_tree_regressor = ExtraTreesRegressor()
extra_tree_regressor.fit(_X_train, _y_reg)
extra_tree_regressor.predict(_X_train[:1])

# Gradient Boosting Classifier
gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_classifier.fit(_X_train, _y_cls)
gradient_boosting_classifier.predict(_X_train[:1])

# Lars
lars = Lars()
lars.fit(_X_train, _y_reg)
lars.predict(_X_train[:1])

# Lasso
lasso = Lasso()
lasso.fit(_X_train, _y_reg)
lasso.predict(_X_train[:1])

# LassoLars
lasso_lars = LassoLars()
lasso_lars.fit(_X_train, _y_reg)
lasso_lars.predict(_X_train[:1])

# LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(_X_train, _y_reg)
linear_regression.predict(_X_train[:1])

# LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(_X_train, _y_cls)
linear_svc.predict(_X_train[:1])

# LogisticRegression
logistic_regression = LogisticRegression(max_iter=200)
logistic_regression.fit(_X_train, _y_cls)
logistic_regression.predict(_X_train[:1])

# Naive Bayes Classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(_X_train, _y_cls)
naive_bayes_classifier.predict(_X_train[:1])

# OrthogonalMatchingPursuit
orthogonal_matching_pursuit = OrthogonalMatchingPursuit()
orthogonal_matching_pursuit.fit(_X_train, _y_reg)
orthogonal_matching_pursuit.predict(_X_train[:1])

# RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(_X_train, _y_cls)
random_forest_classifier.predict(_X_train[:1])

# RandomForestRegressor
random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(_X_train, _y_reg)
random_forest_regressor.predict(_X_train[:1])

# Ridge
ridge = Ridge()
ridge.fit(_X_train, _y_reg)
ridge.predict(_X_train[:1])

# Ridge Classifier
from sklearn.linear_model import RidgeClassifier
ridge_classifier = RidgeClassifier()
ridge_classifier.fit(_X_train, _y_cls)
ridge_classifier.predict(_X_train[:1])

# SGDRegressor
sgd_regressor = SGDRegressor()
sgd_regressor.fit(_X_train, _y_reg)
sgd_regressor.predict(_X_train[:1])

# SVC
svc = SVC()
svc.fit(_X_train, _y_cls)
svc.predict(_X_train[:1])

# SVR
svr = SVR()
svr.fit(_X_train, _y_reg)
svr.predict(_X_train[:1])


# =============================================================================
# 1.  CATBOOST MODELS
# =============================================================================

from catboost import CatBoostClassifier, CatBoostRegressor

# CatBoost Classifier
catboost_classifier = CatBoostClassifier(iterations=10, verbose=0)
catboost_classifier.fit(_X_train, _y_cls)
catboost_classifier.predict(_X_train[:1])

# CatBoost Regressor
catboost_regressor = CatBoostRegressor(iterations=10, verbose=0)
catboost_regressor.fit(_X_train, _y_reg)
catboost_regressor.predict(_X_train[:1])


# =============================================================================
# 2.  LIGHTGBM MODELS
# =============================================================================

import lightgbm as lgb

# LightGBM Classifier
lightgbm_classifier = lgb.LGBMClassifier(n_estimators=10)
lightgbm_classifier.fit(_X_train, _y_cls)
lightgbm_classifier.predict(_X_train[:1])

# LightGBM Regressor
lightgbm_regressor = lgb.LGBMRegressor(n_estimators=10)
lightgbm_regressor.fit(_X_train, _y_reg)
lightgbm_regressor.predict(_X_train[:1])


# =============================================================================
# 3.  XGBOOST MODELS
# =============================================================================

import xgboost as xgb

# XGBoost Classifier
xgboost_classifier = xgb.XGBClassifier(n_estimators=10, verbosity=0)
xgboost_classifier.fit(_X_train, _y_cls)
xgboost_classifier.predict(_X_train[:1])

# XGBoost Regressor
xgboost_regressor = xgb.XGBRegressor(n_estimators=10, verbosity=0)
xgboost_regressor.fit(_X_train, _y_reg)
xgboost_regressor.predict(_X_train[:1])


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


# =============================================================================
# 6.  ANTHROPIC (claude-*) MODELS  – alphabetical
# =============================================================================

import anthropic

_anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def _claude_call(model_name: str) -> anthropic.types.Message:
    return _anthropic_client.messages.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": DUMMY_PROMPT}],
    )

# claude-3-5-sonnet-20240620
resp = _claude_call("claude-3-5-sonnet-20240620")

# claude-3-5-sonnet-20241022
resp = _claude_call("claude-3-5-sonnet-20241022")

# claude-3-7-sonnet-20250219
resp = _claude_call("claude-3-7-sonnet-20250219")

# claude-3-haiku-20240307
resp = _claude_call("claude-3-haiku-20240307")

# claude-haiku-4-5-20251001
resp = _claude_call("claude-haiku-4-5-20251001")

# claude-opus-4-6
resp = _claude_call("claude-opus-4-6")

# claude-sonnet-4-20250514
resp = _claude_call("claude-sonnet-4-20250514")

# claude-sonnet-4-5-20250929
resp = _claude_call("claude-sonnet-4-5-20250929")

# claude-sonnet-4-6
resp = _claude_call("claude-sonnet-4-6")


# =============================================================================
# 7.  GOOGLE GENERATIVE AI (gemini-*) MODELS  – alphabetical
# =============================================================================

import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

def _gemini_call(model_name: str) -> genai.types.GenerateContentResponse:
    model = genai.GenerativeModel(model_name)
    return model.generate_content(DUMMY_PROMPT)

# gemini-1.0-pro
resp = _gemini_call("gemini-1.0-pro")

# gemini-1.5-flash
resp = _gemini_call("gemini-1.5-flash")

# gemini-1.5-pro
resp = _gemini_call("gemini-1.5-pro")

# gemini-2.0-flash
resp = _gemini_call("gemini-2.0-flash")

# gemini-2.5-flash
resp = _gemini_call("gemini-2.5-flash")

# gemini-2.5-pro
resp = _gemini_call("gemini-2.5-pro")

# gemini-3-pro-preview
resp = _gemini_call("gemini-3-pro-preview")


# =============================================================================
# 8.  OPENAI (gpt-* and o<int>-*) MODELS  – alphabetical
# =============================================================================

from openai import OpenAI

_openai_client = OpenAI(api_key=OPENAI_API_KEY)

def _openai_chat(model_name: str) -> openai.types.chat.ChatCompletion:
    return _openai_client.chat.completions.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": DUMMY_PROMPT}],
    )

# gpt-3.5-turbo
resp = _openai_chat("gpt-3.5-turbo")

# gpt-3.5-turbo-16k
resp = _openai_chat("gpt-3.5-turbo-16k")

# gpt-4
resp = _openai_chat("gpt-4")

# gpt-4-turbo
resp = _openai_chat("gpt-4-turbo")

# gpt-4.1
resp = _openai_chat("gpt-4.1")

# gpt-4.1-mini
resp = _openai_chat("gpt-4.1-mini")

# gpt-4.1-nano
resp = _openai_chat("gpt-4.1-nano")

# gpt-4o
resp = _openai_chat("gpt-4o")

# gpt-4o-mini
resp = _openai_chat("gpt-4o-mini")

# gpt-5
resp = _openai_chat("gpt-5")

# gpt-5-mini
resp = _openai_chat("gpt-5-mini")

# gpt-5-nano
resp = _openai_chat("gpt-5-nano")

# o1-mini
resp = _openai_chat("o1-mini")

# o1-pro
resp = _openai_chat("o1-pro")

# o3-mini
resp = _openai_chat("o3-mini")

# o4-mini
resp = _openai_chat("o4-mini")


# =============================================================================
# 9.  HUGGING FACE  (all model-ids that contain '/')  – alphabetical
# =============================================================================

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Helper: generic text-generation pipeline
def _hf_text_gen(model_id: str, **kwargs):
    pipe = pipeline("text-generation", model=model_id, **kwargs)
    return pipe(DUMMY_PROMPT, max_new_tokens=32)

# Helper: generic fill-mask / feature-extraction pipeline
def _hf_pipe(task: str, model_id: str, **kwargs):
    pipe = pipeline(task, model=model_id, **kwargs)
    return pipe(DUMMY_TEXT)

# ── 01-ai ──────────────────────────────────────────────────────────────────
# 01-ai/Yi-34B
_hf_text_gen("01-ai/Yi-34B")

# 01-ai/Yi-34B-Chat
_hf_text_gen("01-ai/Yi-34B-Chat")

# ── allenai ────────────────────────────────────────────────────────────────
# allenai/OLMo-7B
_hf_text_gen("allenai/OLMo-7B")

# allenai/OLMo-7B-Instruct
_hf_text_gen("allenai/OLMo-7B-Instruct")

# allenai/OLMo-7B-Twin-2T
_hf_text_gen("allenai/OLMo-7B-Twin-2T")

# allenai/scibert_scivocab_cased
_hf_pipe("fill-mask", "allenai/scibert_scivocab_cased")

# allenai/scibert_scivocab_uncased
_hf_pipe("fill-mask", "allenai/scibert_scivocab_uncased")

# ── arcee-ai ───────────────────────────────────────────────────────────────
# arcee-ai/saul-zephyr-7b-slerp
_hf_text_gen("arcee-ai/saul-zephyr-7b-slerp")

# ── bigcode ────────────────────────────────────────────────────────────────
# bigcode/santacoder
_hf_text_gen("bigcode/santacoder")

# bigcode/starcoder
_hf_text_gen("bigcode/starcoder")

# ── bigscience ─────────────────────────────────────────────────────────────
# bigscience/bloom-1b1
_hf_text_gen("bigscience/bloom-1b1")

# bigscience/bloom-1b7
_hf_text_gen("bigscience/bloom-1b7")

# bigscience/bloom-3b
_hf_text_gen("bigscience/bloom-3b")

# bigscience/bloom-560m
_hf_text_gen("bigscience/bloom-560m")

# bigscience/bloom-7b1
_hf_text_gen("bigscience/bloom-7b1")

# bigscience/mt0-base
pipe_mt0_base = pipeline("text2text-generation", model="bigscience/mt0-base")
pipe_mt0_base(DUMMY_PROMPT, max_new_tokens=32)

# bigscience/mt0-small
pipe_mt0_small = pipeline("text2text-generation", model="bigscience/mt0-small")
pipe_mt0_small(DUMMY_PROMPT, max_new_tokens=32)

# ── cardiffnlp ─────────────────────────────────────────────────────────────
# cardiffnlp/twitter-roberta-base-sentiment-latest
pipe_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
pipe_sentiment(DUMMY_TEXT)

# ── codellama ──────────────────────────────────────────────────────────────
# codellama/CodeLlama-13b-hf
_hf_text_gen("codellama/CodeLlama-13b-hf")

# codellama/CodeLlama-13b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-13b-Instruct-hf")

# codellama/CodeLlama-34b-hf
_hf_text_gen("codellama/CodeLlama-34b-hf")

# codellama/CodeLlama-34b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-34b-Instruct-hf")

# codellama/CodeLlama-70b-hf
_hf_text_gen("codellama/CodeLlama-70b-hf")

# codellama/CodeLlama-70b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-70b-Instruct-hf")

# codellama/CodeLlama-7b-hf
_hf_text_gen("codellama/CodeLlama-7b-hf")

# codellama/CodeLlama-7b-Instruct-hf
_hf_text_gen("codellama/CodeLlama-7b-Instruct-hf")

# ── CohereForAI ────────────────────────────────────────────────────────────
# CohereForAI/c4ai-command-r7b-12-2024
_hf_text_gen("CohereForAI/c4ai-command-r7b-12-2024")

# ── deepseek-ai ────────────────────────────────────────────────────────────
# deepseek-ai/deepseek-coder-33b-instruct
_hf_text_gen("deepseek-ai/deepseek-coder-33b-instruct")

# deepseek-ai/deepseek-llm-67b-chat
_hf_text_gen("deepseek-ai/deepseek-llm-67b-chat")

# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
_hf_text_gen("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
_hf_text_gen("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

# ── distilbert ─────────────────────────────────────────────────────────────
# distilbert/distilbert-base-uncased
_hf_pipe("fill-mask", "distilbert/distilbert-base-uncased")

# distilbert/distilroberta-base
_hf_pipe("fill-mask", "distilbert/distilroberta-base")

# ── facebook ───────────────────────────────────────────────────────────────
# facebook/bart-large
pipe_bart = pipeline("summarization", model="facebook/bart-large")
pipe_bart("Hugging Face is a technology company.", max_new_tokens=32)

# facebook/blenderbot-400M-distill
pipe_blender = pipeline("conversational", model="facebook/blenderbot-400M-distill")
from transformers import Conversation
pipe_blender(Conversation(DUMMY_PROMPT))

# facebook/regnet-x-016
pipe_regnet_016 = pipeline("image-classification", model="facebook/regnet-x-016")
pipe_regnet_016(DUMMY_IMAGE_PATH)

# facebook/regnet-x-032
pipe_regnet_032 = pipeline("image-classification", model="facebook/regnet-x-032")
pipe_regnet_032(DUMMY_IMAGE_PATH)

# facebook/regnet-x-040
pipe_regnet_040 = pipeline("image-classification", model="facebook/regnet-x-040")
pipe_regnet_040(DUMMY_IMAGE_PATH)

# facebook/regnet-x-064
pipe_regnet_064 = pipeline("image-classification", model="facebook/regnet-x-064")
pipe_regnet_064(DUMMY_IMAGE_PATH)

# facebook/regnet-x-080
pipe_regnet_080 = pipeline("image-classification", model="facebook/regnet-x-080")
pipe_regnet_080(DUMMY_IMAGE_PATH)

# facebook/regnet-x-120
pipe_regnet_120 = pipeline("image-classification", model="facebook/regnet-x-120")
pipe_regnet_120(DUMMY_IMAGE_PATH)

# ── FacebookAI ─────────────────────────────────────────────────────────────
# FacebookAI/roberta-base
_hf_pipe("fill-mask", "FacebookAI/roberta-base")

# FacebookAI/roberta-large
_hf_pipe("fill-mask", "FacebookAI/roberta-large")

# ── google ─────────────────────────────────────────────────────────────────
# google-bert/bert-base-cased
_hf_pipe("fill-mask", "google-bert/bert-base-cased")

# google-bert/bert-base-uncased
_hf_pipe("fill-mask", "google-bert/bert-base-uncased")

# google-bert/bert-large-uncased
_hf_pipe("fill-mask", "google-bert/bert-large-uncased")

# google-t5/t5-base
pipe_t5_base = pipeline("text2text-generation", model="google-t5/t5-base")
pipe_t5_base(DUMMY_PROMPT, max_new_tokens=32)

# google-t5/t5-large
pipe_t5_large = pipeline("text2text-generation", model="google-t5/t5-large")
pipe_t5_large(DUMMY_PROMPT, max_new_tokens=32)

# google-t5/t5-small
pipe_t5_small = pipeline("text2text-generation", model="google-t5/t5-small")
pipe_t5_small(DUMMY_PROMPT, max_new_tokens=32)

# google/efficientnet-b0
pipe_effnet_b0 = pipeline("image-classification", model="google/efficientnet-b0")
pipe_effnet_b0(DUMMY_IMAGE_PATH)

# google/efficientnet-b7
pipe_effnet_b7 = pipeline("image-classification", model="google/efficientnet-b7")
pipe_effnet_b7(DUMMY_IMAGE_PATH)

# google/flan-t5-xxl
pipe_flan = pipeline("text2text-generation", model="google/flan-t5-xxl")
pipe_flan(DUMMY_PROMPT, max_new_tokens=32)

# google/gemma-2-9b
_hf_text_gen("google/gemma-2-9b")

# google/gemma-2-9b-it
_hf_text_gen("google/gemma-2-9b-it")

# google/gemma-2b-it
_hf_text_gen("google/gemma-2b-it")

# google/gemma-3-12b-it
_hf_text_gen("google/gemma-3-12b-it")

# google/gemma-3-27b-it
_hf_text_gen("google/gemma-3-27b-it")

# google/gemma-7b-it
_hf_text_gen("google/gemma-7b-it")

# google/pegasus-large
pipe_pegasus = pipeline("summarization", model="google/pegasus-large")
pipe_pegasus("Hugging Face is a technology company.", max_new_tokens=32)

# ── HuggingFaceM4 ──────────────────────────────────────────────────────────
# HuggingFaceM4/idefics-80b  (multimodal; image+text)
from transformers import IdeficsForVisionText2Text, AutoProcessor
processor_idefics = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-80b")
model_idefics = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-80b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs_idefics = processor_idefics(
    text=[DUMMY_PROMPT], return_tensors="pt"
)
model_idefics.generate(**inputs_idefics, max_new_tokens=32)

# HuggingFaceM4/idefics-9b
processor_idefics_9b = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")
model_idefics_9b = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-9b",
    torch_dtype=torch.bfloat16,
)
inputs_idefics_9b = processor_idefics_9b(
    text=[DUMMY_PROMPT], return_tensors="pt"
)
model_idefics_9b.generate(**inputs_idefics_9b, max_new_tokens=32)

# HuggingFaceM4/idefics-9b-instruct
processor_idefics_9b_inst = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct"
)
model_idefics_9b_inst = IdeficsForVisionText2Text.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct",
    torch_dtype=torch.bfloat16,
)
inputs_idefics_9b_inst = processor_idefics_9b_inst(
    text=[DUMMY_PROMPT], return_tensors="pt"
)
model_idefics_9b_inst.generate(**inputs_idefics_9b_inst, max_new_tokens=32)

# ── j-hartmann ─────────────────────────────────────────────────────────────
# j-hartmann/emotion-english-distilroberta-base
pipe_emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
)
pipe_emotion(DUMMY_TEXT)

# ── llava-hf ───────────────────────────────────────────────────────────────
# llava-hf/llava-1.5-13b-hf
from transformers import LlavaForConditionalGeneration, AutoProcessor as LlavaProcessor
processor_llava = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
model_llava = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16
)
inputs_llava = processor_llava(
    text=DUMMY_PROMPT, return_tensors="pt"
)
model_llava.generate(**inputs_llava, max_new_tokens=32)

# llava/hf_llava-v1.6-mistral-7b-hf
processor_llava16 = LlavaProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
model_llava16 = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16
)
inputs_llava16 = processor_llava16(
    text=DUMMY_PROMPT, return_tensors="pt"
)
model_llava16.generate(**inputs_llava16, max_new_tokens=32)

# ── lmsys ──────────────────────────────────────────────────────────────────
# lmsys/vicuna-13b-v1.5
_hf_text_gen("lmsys/vicuna-13b-v1.5")

# lmsys/vicuna-7b-v1.5
_hf_text_gen("lmsys/vicuna-7b-v1.5")

# ── meta-llama ─────────────────────────────────────────────────────────────
# meta-llama/Llama-2-13b-chat-hf
_hf_text_gen("meta-llama/Llama-2-13b-chat-hf")

# meta-llama/Llama-2-70b-chat-hf
_hf_text_gen("meta-llama/Llama-2-70b-chat-hf")

# meta-llama/Llama-2-7b-chat-hf
_hf_text_gen("meta-llama/Llama-2-7b-chat-hf")

# meta-llama/Llama-3.1-405B-Instruct-FP8
_hf_text_gen("meta-llama/Llama-3.1-405B-Instruct-FP8")

# meta-llama/Llama-3.1-70B
_hf_text_gen("meta-llama/Llama-3.1-70B")

# meta-llama/Llama-3.1-70B-Instruct
_hf_text_gen("meta-llama/Llama-3.1-70B-Instruct")

# meta-llama/Llama-3.2-1B
_hf_text_gen("meta-llama/Llama-3.2-1B")

# meta-llama/Llama-3.2-1B-Instruct
_hf_text_gen("meta-llama/Llama-3.2-1B-Instruct")

# meta-llama/Llama-3.2-3B
_hf_text_gen("meta-llama/Llama-3.2-3B")

# meta-llama/Llama-3.2-3B-Instruct
_hf_text_gen("meta-llama/Llama-3.2-3B-Instruct")

# meta-llama/Llama-3.3-70B-Instruct
_hf_text_gen("meta-llama/Llama-3.3-70B-Instruct")

# meta-llama/Llama-4-Scout-17B-16E-Instruct
_hf_text_gen("meta-llama/Llama-4-Scout-17B-16E-Instruct")

# meta-llama/Meta-Llama-3-70B
_hf_text_gen("meta-llama/Meta-Llama-3-70B")

# meta-llama/Meta-Llama-3-70B-Instruct
_hf_text_gen("meta-llama/Meta-Llama-3-70B-Instruct")

# meta-llama/Meta-Llama-3-8B
_hf_text_gen("meta-llama/Meta-Llama-3-8B")

# meta-llama/Meta-Llama-3-8B-Instruct
_hf_text_gen("meta-llama/Meta-Llama-3-8B-Instruct")

# meta-llama/Meta-Llama-3.1-8B
_hf_text_gen("meta-llama/Meta-Llama-3.1-8B")

# meta-llama/Meta-Llama-3.1-8B-Instruct
_hf_text_gen("meta-llama/Meta-Llama-3.1-8B-Instruct")

# ── microsoft ──────────────────────────────────────────────────────────────
# microsoft/deberta-base
_hf_pipe("fill-mask", "microsoft/deberta-base")

# microsoft/deberta-v3-base
_hf_pipe("fill-mask", "microsoft/deberta-v3-base")

# microsoft/DialoGPT-medium
pipe_dialogpt = pipeline("text-generation", model="microsoft/DialoGPT-medium")
pipe_dialogpt(DUMMY_PROMPT, max_new_tokens=32)

# microsoft/focalnet-tiny-lrf
pipe_focalnet = pipeline("image-classification", model="microsoft/focalnet-tiny-lrf")
pipe_focalnet(DUMMY_IMAGE_PATH)

# microsoft/Phi-3-medium-4k-instruct
_hf_text_gen("microsoft/Phi-3-medium-4k-instruct")

# microsoft/Phi-3-mini-4k-instruct
_hf_text_gen("microsoft/Phi-3-mini-4k-instruct")

# microsoft/phi-4
_hf_text_gen("microsoft/phi-4")

# microsoft/resnet-101
pipe_ms_rn101 = pipeline("image-classification", model="microsoft/resnet-101")
pipe_ms_rn101(DUMMY_IMAGE_PATH)

# microsoft/resnet-152
pipe_ms_rn152 = pipeline("image-classification", model="microsoft/resnet-152")
pipe_ms_rn152(DUMMY_IMAGE_PATH)

# microsoft/resnet-18
pipe_ms_rn18 = pipeline("image-classification", model="microsoft/resnet-18")
pipe_ms_rn18(DUMMY_IMAGE_PATH)

# microsoft/resnet-50
pipe_ms_rn50 = pipeline("image-classification", model="microsoft/resnet-50")
pipe_ms_rn50(DUMMY_IMAGE_PATH)

# ── mistralai ──────────────────────────────────────────────────────────────
# mistralai/Codestral-22B-v0.1
_hf_text_gen("mistralai/Codestral-22B-v0.1")

# mistralai/Ministral-8B-Instruct-2410
_hf_text_gen("mistralai/Ministral-8B-Instruct-2410")

# mistralai/Mistral-7B-Instruct-v0.1
_hf_text_gen("mistralai/Mistral-7B-Instruct-v0.1")

# mistralai/Mistral-7B-Instruct-v0.2
_hf_text_gen("mistralai/Mistral-7B-Instruct-v0.2")

# mistralai/Mistral-7B-Instruct-v0.3
_hf_text_gen("mistralai/Mistral-7B-Instruct-v0.3")

# mistralai/Mistral-7B-v0.3
_hf_text_gen("mistralai/Mistral-7B-v0.3")

# mistralai/Mistral-Large-Instruct-2407
_hf_text_gen("mistralai/Mistral-Large-Instruct-2407")

# mistralai/Mistral-Small-3.2-24B-Instruct-2506
_hf_text_gen("mistralai/Mistral-Small-3.2-24B-Instruct-2506")

# mistralai/Mixtral-8x22B-Instruct-v0.1
_hf_text_gen("mistralai/Mixtral-8x22B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-Instruct-v0.1
_hf_text_gen("mistralai/Mixtral-8x7B-Instruct-v0.1")

# mistralai/Mixtral-8x7B-v0.1
_hf_text_gen("mistralai/Mixtral-8x7B-v0.1")

# ── NousResearch ───────────────────────────────────────────────────────────
# NousResearch/Nous-Capybara-7B-V1.9
_hf_text_gen("NousResearch/Nous-Capybara-7B-V1.9")

# NousResearch/Nous-Hermes-llama-2-7b
_hf_text_gen("NousResearch/Nous-Hermes-llama-2-7b")

# NousResearch/Nous-Hermes-Llama2-13b
_hf_text_gen("NousResearch/Nous-Hermes-Llama2-13b")

# ── nreimers ───────────────────────────────────────────────────────────────
# nreimers/BERT-Tiny_L-2_H-128_A-2
_hf_pipe("fill-mask", "nreimers/BERT-Tiny_L-2_H-128_A-2")

# nreimers/MiniLM-L6-H384-uncased
_hf_pipe("feature-extraction", "nreimers/MiniLM-L6-H384-uncased")

# ── NumbersStation ─────────────────────────────────────────────────────────
# NumbersStation/nsql-llama-2-7B
_hf_text_gen("NumbersStation/nsql-llama-2-7B")

# ── nvidia ─────────────────────────────────────────────────────────────────
# nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
_hf_text_gen("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")

# nvidia/parakeet-tdt-0.6b-v3  (ASR model)
pipe_parakeet = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")
# pipe_parakeet("audio.wav")  # provide a real audio file at runtime

# ── Open-Orca ──────────────────────────────────────────────────────────────
# Open-Orca/Mistral-7B-OpenOrca
_hf_text_gen("Open-Orca/Mistral-7B-OpenOrca")

# ── openai (HF namespace) ──────────────────────────────────────────────────
# openai/gpt-oss-120b
_hf_text_gen("openai/gpt-oss-120b")

# openai/gpt-oss-20b
_hf_text_gen("openai/gpt-oss-20b")

# openai/whisper-base  (ASR)
pipe_whisper_base = pipeline("automatic-speech-recognition", model="openai/whisper-base")
# pipe_whisper_base("audio.wav")

# openai/whisper-large-v3-turbo
pipe_whisper_large = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3-turbo"
)
# pipe_whisper_large("audio.wav")

# ── openchat ───────────────────────────────────────────────────────────────
# openchat/openchat-3.5-1210
_hf_text_gen("openchat/openchat-3.5-1210")

# ── prajjwal1 ──────────────────────────────────────────────────────────────
# prajjwal1/bert-tiny
_hf_pipe("fill-mask", "prajjwal1/bert-tiny")

# ── Qwen ───────────────────────────────────────────────────────────────────
# Qwen/Qwen1.5-0.5B-Chat
_hf_text_gen("Qwen/Qwen1.5-0.5B-Chat")

# Qwen/Qwen1.5-1.8B-Chat
_hf_text_gen("Qwen/Qwen1.5-1.8B-Chat")

# Qwen/Qwen1.5-110B-Chat
_hf_text_gen("Qwen/Qwen1.5-110B-Chat")

# Qwen/Qwen1.5-14B-Chat
_hf_text_gen("Qwen/Qwen1.5-14B-Chat")

# Qwen/Qwen1.5-32B-Chat
_hf_text_gen("Qwen/Qwen1.5-32B-Chat")

# Qwen/Qwen1.5-4B-Chat
_hf_text_gen("Qwen/Qwen1.5-4B-Chat")

# Qwen/Qwen1.5-72B-Chat
_hf_text_gen("Qwen/Qwen1.5-72B-Chat")

# Qwen/Qwen1.5-7B-Chat
_hf_text_gen("Qwen/Qwen1.5-7B-Chat")

# Qwen/Qwen2-72B-Instruct
_hf_text_gen("Qwen/Qwen2-72B-Instruct")

# Qwen/Qwen2-7B-Instruct
_hf_text_gen("Qwen/Qwen2-7B-Instruct")

# Qwen/Qwen2.5-32B
_hf_text_gen("Qwen/Qwen2.5-32B")

# Qwen/Qwen3-4B
_hf_text_gen("Qwen/Qwen3-4B")

# Qwen/Qwen3-Next-80B-A3B-Instruct
_hf_text_gen("Qwen/Qwen3-Next-80B-A3B-Instruct")

# Qwen/QwQ-32B
_hf_text_gen("Qwen/QwQ-32B")

# ── stabilityai ────────────────────────────────────────────────────────────
# stabilityai/stablelm-tuned-alpha-3b
_hf_text_gen("stabilityai/stablelm-tuned-alpha-3b")

# ── tiiuae ─────────────────────────────────────────────────────────────────
# tiiuae/falcon-7b-instruct
_hf_text_gen("tiiuae/falcon-7b-instruct")

# ── timm (namespace/repo models) ───────────────────────────────────────────
# timm/eva_giant_patch14_224.clip_ft_in1k
model_eva = timm.create_model(
    "timm/eva_giant_patch14_224.clip_ft_in1k", pretrained=False
)
model_eva.eval()
model_eva(_dummy_tensor)

# timm/inception_v4.tf_in1k
model_iv4 = timm.create_model("timm/inception_v4.tf_in1k", pretrained=False)
model_iv4.eval()
model_iv4(torch.randn(1, 3, 299, 299))

# timm/levit_128s.fb_dist_in1k
model_levit = timm.create_model("timm/levit_128s.fb_dist_in1k", pretrained=False)
model_levit.eval()
model_levit(_dummy_tensor)

# timm/tf_efficientnet_lite0.in1k
model_effnetlite = timm.create_model(
    "timm/tf_efficientnet_lite0.in1k", pretrained=False
)
model_effnetlite.eval()
model_effnetlite(_dummy_tensor)

# ── TinyLlama ──────────────────────────────────────────────────────────────
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
_hf_text_gen("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ── togethercomputer ───────────────────────────────────────────────────────
# togethercomputer/RedPajama-INCITE-7B-Chat
_hf_text_gen("togethercomputer/RedPajama-INCITE-7B-Chat")

# togethercomputer/RedPajama-INCITE-Chat-3B-v1
_hf_text_gen("togethercomputer/RedPajama-INCITE-Chat-3B-v1")

# ── unsloth ────────────────────────────────────────────────────────────────
# unsloth/DeepSeek-R1-GGUF  – use llama-cpp-python for GGUF models
from llama_cpp import Llama
llm_gguf = Llama.from_pretrained(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    filename="*Q4_K_M.gguf",    # pick a quantisation file
)
llm_gguf(DUMMY_PROMPT, max_tokens=32)