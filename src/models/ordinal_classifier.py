from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

LABEL_ORDER = ["A", "B", "C", "D", "E"]

class FrankHallOrdinalClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.array(LABEL_ORDER)
        self.k_ = len(self.classes_)
        self.estimators_ = []
        
        for i in range(self.k_ - 1):
            y_mapped = np.array([LABEL_ORDER.index(val) for val in y])
            y_binary = (y_mapped > i).astype(int)
            
            clf = clone(self.estimator)
            clf.fit(X, y_binary)
            self.estimators_.append(clf)
            
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        p_greater = np.zeros((n_samples, self.k_ - 1))
        for i, clf in enumerate(self.estimators_):
            p_greater[:, i] = clf.predict_proba(X)[:, 1]
            
        probs = np.zeros((n_samples, self.k_))
        probs[:, 0] = 1.0 - p_greater[:, 0]
        for i in range(1, self.k_ - 1):
            probs[:, i] = p_greater[:, i-1] - p_greater[:, i]
        probs[:, self.k_ - 1] = p_greater[:, self.k_ - 2]
        
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
