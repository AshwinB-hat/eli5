import lightgbm  
import numpy as np 
from eli5.sklearn.utils import handle_vec, get_X, get_X0, add_intercept, predict_proba
from eli5._feature_importances import get_feature_importance_explanation

DESCRIPTION_LIGHTGBM = """
LightGBM feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""


def explain_shap_prediction_lightgbm(
        lgb, 
        doc,
        vec=None,
        top=None,
        top_targets=None,
        target_names=None,
        targets=None,
        feature_names=None,
        feature_re=None,
        feature_filter=None,
        vectorized=False,
        ):
    vec, feature_names = handle_vec(lgb, doc, vec, vectorized, feature_names)
    feature_names.add_feature('<BIAS>')
    lgb_feature_names = lgb.booster_.feature_name()
    coef = _get_lgb_shap_importances(lgb, doc)
    return get_feature_importance_explanation(lgb, vec, coef,
        feature_names=feature_names,
        estimator_feature_names=lgb_feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        description=DESCRIPTION_LIGHTGBM,
        num_features=coef.shape[-1],
        is_regression=isinstance(lgb, lightgbm.LGBMRegressor),
    )

def _get_lgb_shap_importances(lgb, doc):
    coef = lgb.booster_.predict(doc, pred_contrib=True)
    shap_weights = np.sum(coef, axis=0)/coef.shape[0]
    return shap_weights