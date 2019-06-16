from eli5._feature_weights import get_top_features_filtered
from eli5.base import Explanation, TargetExplanation
from eli5._feature_importances import get_feature_importance_explanation

DECISION_PATHS_DESCRIPTION = """
Feature weights are calculated by following decision paths in trees
of an ensemble. Each leaf has an output score, and expected scores can also be
assigned to parent nodes. Contribution of one feature on the decision path
is how much expected score changes from parent to child. Weights of all 
features sum to the output score of the estimator.
"""


DECISION_PATHS_CAVEATS = """
Caveats:
1. Feature weights just show if the feature contributed positively or
   negatively to the final score, and does not show how increasing or
   decreasing the feature value will change the prediction.
2. In some cases, feature weight can be close to zero for an important feature.
   For example, in a single tree that computes XOR function, the feature at the
   top of the tree will have zero weight because expected scores for both
   branches are equal, so decision at the top feature does not change the
   expected score. For an ensemble predicting XOR functions it might not be
   a problem, but it is not reliable if most trees happen to choose the same
   feature at the top.
"""


DECISION_PATHS_EXPLANATION = "".join([
    DECISION_PATHS_DESCRIPTION,
    DECISION_PATHS_CAVEATS
])


DESCRIPTION_CLF_MULTICLASS = """
Features with largest coefficients per class.
""" + DECISION_PATHS_EXPLANATION

DESCRIPTION_CLF_BINARY = """
Features with largest coefficients.
""" + DECISION_PATHS_EXPLANATION

DESCRIPTION_REGRESSION = DESCRIPTION_CLF_BINARY



def get_shap_explanation_summary(estimator, vec, coef,
                                  feature_names,
                                  feature_filter, 
                                  feature_re, top, 
                                  is_regression, Description, is_multiclass=False):
    if(len(coef)==1):
        return get_feature_importance_explanation(estimator, vec, coef,
                                              feature_names=feature_names,
                                              estimator_feature_names=feature_names,
                                              feature_filter=None,
                                              feature_re=None,
                                              top=top,
                                              description=Description,
                                              is_regression=is_regression,
                                              num_features=coef.shape[-1]
                                              )
    else:
        explanation = Explanation(
        estimator=repr(estimator),
        method='shap Explanation',
        description=DESCRIPTION_CLF_MULTICLASS,
        is_regression=is_regression,
        targets=[],
        )
        assert explanation.targets is not None

        for i in range(coef.shape[0]):
            target_expl = TargetExplanation(
                target=feature_names,
                feature_weights=coef[i,:],
                score=None,
                proba=None,
            )
            explanation.targets.append(target_expl)
        return explanation
