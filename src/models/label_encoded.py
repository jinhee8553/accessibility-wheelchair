from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapter for estimators that require integer class labels."""

    _estimator_type = "classifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        encoded_target = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded_target)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        encoded_prediction = self.estimator_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded_prediction)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_
