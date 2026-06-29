"""
02_boosting.py
==============
CatBoost, LightGBM, and XGBoost model calls — alphabetical.

Install:
    pip install catboost lightgbm xgboost numpy
"""

import numpy as np

_X = np.random.rand(20, 4)
_y_cls = np.random.randint(0, 2, 20)
_y_reg = np.random.rand(20)

# ── CatBoost ──────────────────────────────────────────────────────────────
from catboost import CatBoostClassifier, CatBoostRegressor

# CatBoost Classifier
catboost_classifier = CatBoostClassifier(iterations=10, verbose=0)
catboost_classifier.fit(_X, _y_cls)
catboost_classifier.predict(_X[:1])

# CatBoost Regressor
catboost_regressor = CatBoostRegressor(iterations=10, verbose=0)
catboost_regressor.fit(_X, _y_reg)
catboost_regressor.predict(_X[:1])

# ── LightGBM ──────────────────────────────────────────────────────────────
from lightgbm import LightGBMClassifier, LightGBMRegressor
# LightGBM Classifier
lightgbm_classifier = LightGBMClassifier(n_estimators=10)
lightgbm_classifier.fit(_X, _y_cls)
lightgbm_classifier.predict(_X[:1])

# LightGBM Regressor
lightgbm_regressor = LightGBMRegressor(n_estimators=10)
lightgbm_regressor.fit(_X, _y_reg)
lightgbm_regressor.predict(_X[:1])

# ── XGBoost ───────────────────────────────────────────────────────────────
from xgboost import XGBoostClassifier, XGBoostRegressor

# XGBoost Classifier
xgboost_classifier = XGBClassifier(n_estimators=10, verbosity=0)
xgboost_classifier.fit(_X, _y_cls)
xgboost_classifier.predict(_X[:1])

# XGBoost Regressor
xgboost_regressor = XGBoostRegressor(n_estimators=10, verbosity=0)
xgboost_regressor.fit(_X, _y_reg)
xgboost_regressor.predict(_X[:1])
