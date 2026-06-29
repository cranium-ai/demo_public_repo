"""
01_sklearn.py
=============
scikit-learn model calls — alphabetical.

Install:
    pip install scikit-learn numpy
"""

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
    RidgeClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

_X = np.random.rand(20, 4)
_y_cls = np.random.randint(0, 2, 20)
_y_reg = np.random.rand(20)

# BayesianRidge
bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(_X, _y_reg)
bayesian_ridge.predict(_X[:1])

# DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(_X, _y_cls)
decision_tree_classifier.predict(_X[:1])

# DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(_X, _y_reg)
decision_tree_regressor.predict(_X[:1])

# ElasticNet
elastic_net = ElasticNet()
elastic_net.fit(_X, _y_reg)
elastic_net.predict(_X[:1])

# Extra Tree Classifier
extra_tree_classifier = ExtraTreesClassifier()
extra_tree_classifier.fit(_X, _y_cls)
extra_tree_classifier.predict(_X[:1])

# Extra Tree Regressor
extra_tree_regressor = ExtraTreesRegressor()
extra_tree_regressor.fit(_X, _y_reg)
extra_tree_regressor.predict(_X[:1])

# Gradient Boosting Classifier
gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_classifier.fit(_X, _y_cls)
gradient_boosting_classifier.predict(_X[:1])

# Lars
lars = Lars()
lars.fit(_X, _y_reg)
lars.predict(_X[:1])

# Lasso
lasso = Lasso()
lasso.fit(_X, _y_reg)
lasso.predict(_X[:1])

# LassoLars
lasso_lars = LassoLars()
lasso_lars.fit(_X, _y_reg)
lasso_lars.predict(_X[:1])

# LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(_X, _y_reg)
linear_regression.predict(_X[:1])

# LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(_X, _y_cls)
linear_svc.predict(_X[:1])

# LogisticRegression
logistic_regression = LogisticRegression(max_iter=200)
logistic_regression.fit(_X, _y_cls)
logistic_regression.predict(_X[:1])

# Naive Bayes Classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(_X, _y_cls)
naive_bayes_classifier.predict(_X[:1])

# OrthogonalMatchingPursuit
orthogonal_matching_pursuit = OrthogonalMatchingPursuit()
orthogonal_matching_pursuit.fit(_X, _y_reg)
orthogonal_matching_pursuit.predict(_X[:1])

# RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(_X, _y_cls)
random_forest_classifier.predict(_X[:1])

# RandomForestRegressor
random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(_X, _y_reg)
random_forest_regressor.predict(_X[:1])

# Ridge
ridge = Ridge()
ridge.fit(_X, _y_reg)
ridge.predict(_X[:1])

# Ridge Classifier
ridge_classifier = RidgeClassifier()
ridge_classifier.fit(_X, _y_cls)
ridge_classifier.predict(_X[:1])

# SGDRegressor
sgd_regressor = SGDRegressor()
sgd_regressor.fit(_X, _y_reg)
sgd_regressor.predict(_X[:1])

# SVC
svc = SVC()
svc.fit(_X, _y_cls)
svc.predict(_X[:1])

# SVR
svr = SVR()
svr.fit(_X, _y_reg)
svr.predict(_X[:1])
