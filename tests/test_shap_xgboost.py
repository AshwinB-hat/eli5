import pytest 
import xgboost
from sklearn.feature_extraction.text import CountVectorizer


from xgboost import (
    XGBRegressor,
    XGBClassifier
)

from .utils import (
    get_all_features
)
from eli5.shap_xgboost import (
    explain_shap_prediction_xgboost
)

from .utils import format_as_all
def test_explain_xgboost_regressor(boston_train):
    xs, ys, feature_names = boston_train
    reg = XGBRegressor()
    reg.fit(xs, ys)
    res = explain_shap_prediction_xgboost(reg,xs)
    for expl in format_as_all(res, reg):
        assert 'x12' in expl
    res = explain_shap_prediction_xgboost(reg, xs, feature_names=feature_names)
    for expl in format_as_all(res, reg):
        assert 'LSTAT' in expl


def test_explain_xgboost_booster(boston_train):
    xs, ys, feature_names = boston_train
    booster = xgboost.train(
        params={'objective': 'reg:linear', 'silent': True},
        dtrain=xgboost.DMatrix(xs, label=ys),
    )
    res = explain_shap_prediction_xgboost(booster, xs)
    for expl in format_as_all(res, booster):
        assert 'x12' in expl
    res = explain_shap_prediction_xgboost(booster, xs, feature_names=feature_names)
    for expl in format_as_all(res, booster):
        assert 'LSTAT' in expl

def test_explain_xgboost_classifier(
        newsgroups_train):
    docs, ys, target_names = newsgroups_train
    vec = CountVectorizer(stop_words='english')
    xs = vec.fit_transform(docs)
    clf = XGBClassifier(n_estimators=100, max_depth=2)
    clf.fit(xs, ys)
    doc='computer graphics in space: a new religion'
    res = explain_shap_prediction_xgboost(clf, doc, vec=vec, vectorized=False)
    format_as_all(res, clf)
    graphics_weights = res.targets[1].feature_weights
    assert 'computer' in get_all_features(graphics_weights.pos)
    religion_weights = res.targets[3].feature_weights
    assert 'religion' in get_all_features(religion_weights.pos)
