from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted_ = True # IMPORTANT!!!
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs > self.threshold).astype(int)