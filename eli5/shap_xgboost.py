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


DESCRIPTION_XGBOOST_SHAP = """
XGBoost feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

def explain_shap_prediction_xgboost(
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
    
    booster, is_regression = _check_booster_args(xgb)
    xgb_feature_names = booster.feature_names
    vec, feature_names = handle_vec(
        xgb, doc, vec, vectorized, feature_names,
        num_features=len(xgb_feature_names))
    
    #adding the intercept to the feature names. 
    feature_names.add_feature('<BIAS>')
    
    if missing is None:
        missing = np.nan if isinstance(xgb, Booster) else xgb.missing
    
    if str(type(doc)).endswith("xgboost.core.DMatrix'>"):
        dmatrix = doc
    else:
        if(vec is not None):
            dmatrix=DMatrix(get_X(doc,vec=vec,vectorized=vectorized), missing=missing)
        else:
            dmatrix = DMatrix(doc, missing=missing)
    
    n_targets = dmatrix.num_col()
    n_row = dmatrix.num_row()
    # score_weights = _prediction_feature_weights(
    #     booster, dmatrix, n_targets, feature_names, xgb_feature_names)
    coef = _xgb_feature_importances(booster, dmatrix)
    return get_feature_importance_explanation(xgb, vec, coef,
                                              feature_names=feature_names,
                                              estimator_feature_names=xgb_feature_names,
                                              feature_filter=None,
                                              feature_re=None,
                                              top=top,
                                              description=DESCRIPTION_XGBOOST_SHAP,
                                              is_regression=is_regression,
                                              num_features=coef.shape[-1]
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

def _xgb_feature_importances(booster, dMatrix):
    fs = booster.predict(dMatrix, pred_contribs=True)
    all_features = np.sum(fs, axis=0)
    return all_features / fs.shape[0]
    
def _missing_values_set_to_nan(values, missing_value, sparse_missing):
    """ Return a copy of values where missing values (equal to missing_value)
    are replaced to nan according. If sparse_missing is True,
    entries missing in a sparse matrix will also be set to nan.
    Sparse matrices will be converted to dense format.
    """
    if sp.issparse(values):
        assert values.shape[0] == 1
    if sparse_missing and sp.issparse(values) and missing_value != 0:
        # Nothing special needs to be done for missing.value == 0 because
        # missing values are assumed to be zero in sparse matrices.
        values_coo = values.tocoo()
        values = values.toarray()[0]
        missing_mask = values == 0
        # fix for possible zero values
        missing_mask[values_coo.col] = False
        values[missing_mask] = np.nan
    elif is_sparse_vector(values):
        values = values.toarray()[0]
    else:
        values = values.copy()
    if not np.isnan(missing_value):
        values[values == missing_value] = np.nan
    return values
