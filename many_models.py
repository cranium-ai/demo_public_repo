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