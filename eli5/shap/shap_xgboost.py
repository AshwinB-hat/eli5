# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial
import re
from typing import Any, Dict, List, Tuple, Optional, Pattern

import numpy as np  # type: ignore
import scipy.sparse as sp  # type: ignore
from xgboost import (  # type: ignore
    XGBClassifier,
    XGBRegressor,
    Booster,
    DMatrix
)

from eli5.explain import explain_weights, explain_prediction
from eli5.sklearn.utils import (
    add_intercept,
    get_X,
    get_X0,
    handle_vec,
    predict_proba
)
from eli5.utils import is_sparse_vector
from eli5._decision_path import get_decision_path_explanation
from eli5._feature_importances import get_feature_importance_explanation


DESCRIPTION_XGBOOST = """
XGBoost feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

def explain_prediction_xgboost(
        xgb, doc,
        vec=None,
        top=None,
        top_targets=None,
        target_names=None,
        targets=None,
        feature_names=None,
        feature_re=None,  # type: Pattern[str]
        feature_filter=None,
        vectorized=False,  # type: bool
        is_regression=None,  # type: bool
        missing=None,  # type: bool
        ):
    """ Return an explanation of XGBoost prediction (via scikit-learn wrapper
    XGBClassifier or XGBRegressor, or via xgboost.Booster) as feature weights.

    See :func:`eli5.explain_prediction` for description of
    ``top``, ``top_targets``, ``target_names``, ``targets``,
    ``feature_names``, ``feature_re`` and ``feature_filter`` parameters.

    Parameters
    ----------
    vec : vectorizer, optional
        A vectorizer instance used to transform
        raw features to the input of the estimator ``xgb``
        (e.g. a fitted CountVectorizer instance); you can pass it
        instead of ``feature_names``.

    vectorized : bool, optional
        A flag which tells eli5 if ``doc`` should be
        passed through ``vec`` or not. By default it is False, meaning that
        if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
        estimator. Set it to True if you're passing ``vec``,
        but ``doc`` is already vectorized.

    is_regression : bool, optional
        Pass if an ``xgboost.Booster`` is passed as the first argument.
        True if solving a regression problem ("objective" starts with "reg")
        and False for a classification problem.
        If not set, regression is assumed for a single target estimator
        and proba will not be shown.

    missing : optional
        Pass if an ``xgboost.Booster`` is passed as the first argument.
        Set it to the same value as the ``missing`` argument to
        ``xgboost.DMatrix``.
        Matters only if sparse values are used. Default is ``np.nan``.

    Method for determining feature importances follows an idea from
    http://blog.datadive.net/interpreting-random-forests/.
    Feature weights are calculated by following decision paths in trees
    of an ensemble.
    Each leaf has an output score, and expected scores can also be assigned
    to parent nodes.
    Contribution of one feature on the decision path is how much expected score
    changes from parent to child.
    Weights of all features sum to the output score of the estimator.
    """
    booster, is_regression = _check_booster_args(xgb, is_regression)
    xgb_feature_names = booster.feature_names
    vec, feature_names = handle_vec(
        xgb, doc, vec, vectorized, feature_names,
        num_features=len(xgb_feature_names))
    if feature_names.bias_name is None:
        # XGBoost estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'

    X = get_X(doc, vec, vectorized=vectorized)
    if sp.issparse(X):
        # Work around XGBoost issue:
        # https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
        X = X.tocsc()

    if missing is None:
        missing = np.nan if isinstance(xgb, Booster) else xgb.missing
    dmatrix = DMatrix(X, missing=missing)

    if isinstance(xgb, Booster):
        prediction = xgb.predict(dmatrix)
        n_targets = prediction.shape[-1]  # type: int
        if is_regression is None:
            # When n_targets is 1, this can be classification too,
            # but it's safer to assume regression.
            # If n_targets > 1, it must be classification.
            is_regression = n_targets == 1
        if is_regression:
            proba = None
        else:
            if n_targets == 1:
                p, = prediction
                proba = np.array([1 - p, p])
            else:
                proba, = prediction
    else:
        proba = predict_proba(xgb, X)
        n_targets = _xgb_n_targets(xgb)

    if is_regression:
        names = ['y']
    elif isinstance(xgb, Booster):
        names = np.arange(max(2, n_targets))
    else:
        names = xgb.classes_

    scores_weights = _prediction_feature_weights(
        booster, dmatrix, n_targets, feature_names, xgb_feature_names)

    x = get_X0(add_intercept(X))
    x = _missing_values_set_to_nan(x, missing, sparse_missing=True)

    return get_decision_path_explanation(
        xgb, doc, vec,
        x=x,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        vectorized=vectorized,
        original_display_names=names,
        target_names=target_names,
        targets=targets,
        top_targets=top_targets,
        is_regression=is_regression,
        is_multiclass=n_targets > 1,
        proba=proba,
        get_score_weights=lambda label_id: scores_weights[label_id],
     )

def _check_booster_args(xgb, is_regression=None):
    # type: (Any, Optional[bool]) -> Tuple[Booster, Optional[bool]]
    if isinstance(xgb, Booster):
        booster = xgb
    else:
        if hasattr(xgb, 'get_booster'):
            booster = xgb.get_booster()
        else:  # xgb < 0.7
            booster = xgb.booster()
        _is_regression = isinstance(xgb, XGBRegressor)
        if is_regression is not None and is_regression != _is_regression:
            raise ValueError(
                'Inconsistent is_regression={} passed. '
                'You don\'t have to pass it when using scikit-learn API'
                .format(is_regression))
        is_regression = _is_regression
    return booster, is_regression

def _prediction_feature_weights(booster, dmatrix, n_targets,
                                feature_names, xgb_feature_names):
    """ For each target, return score and numpy array with feature weights
    on this prediction, following an idea from
    http://blog.datadive.net/interpreting-random-forests/
    """
    # XGBClassifier does not have pred_leaf argument, so use booster
    phi = booster.predict(dmatrix, pred_contrib=True)
    xgb_feature_names = {f: i for i, f in enumerate(xgb_feature_names)}
    
    target_feature_weights = partial(
        _target_feature_weights,
        feature_names=feature_names, xgb_feature_names=xgb_feature_names)
    # if n_targets > 1:
    #     # For multiclass, XGBoost stores dumps and leaf_ids in a 1d array,
    #     # so we need to split them.
    #     scores_weights = [
    #         target_feature_weights(
    #             leaf_ids[target_idx::n_targets],
    #             tree_dumps[target_idx::n_targets],
    #         ) for target_idx in range(n_targets)]
    # else:
    scores_weights = [target_feature_weights(phi)]
    return scores_weights

def _target_feature_weights(phi, feature_names,
                            xgb_feature_names):
    feature_weights = np.zeros(len(feature_names))
    # All trees in XGBoost give equal contribution to the prediction:
    # it is equal to sum of "leaf" values in leafs
    # before applying loss-specific function
    # (e.g. logistic for "binary:logistic" loss).
    return feature_weights