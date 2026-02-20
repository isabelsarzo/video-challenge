from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
import numpy as np

class Model(BaseEstimator):
    """
    Wrapper for classifier in pipeline.

    NOTE: This wrapper was implemented for more flexibility.

    Attributes:
        classifier (BaseEstimator): 
            A scikit-learn model instance (e.g. MLPClassifier(), XGBClassifier()) 
            
    """

    def __init__(self, classifier):
        """
        Initializes Model.

        Args:
            classifier (BaseEstimator): 
                A scikit-learn model instance (e.g. MLPClassifier(), XGBClassifier()).
        """
        self.classifier = classifier

    def fit(self, X, y, **kwargs):
        """
        Fits the model stored in 'classifier' to the training data.

        Args:
            X (array-like of shape (n_samples, n_features)): The training input samples
            y (array-like of shape (n_samples,)): Target labels

        Returns:
            self: The fitted model instance

        """
        if isinstance(self.classifier, XGBClassifier):
            neg, pos = np.bincount(y)
            spw = int(neg / pos)
            self.classifier.set_params(scale_pos_weight=spw)
       
        return self.classifier.fit(X, y, **kwargs)
    
    def predict(self, X):
        """
        Predicts class labels for the given input data.

        Args:
            X (array-like of shape (n_samples, n_features)): The input samples

        Returns:
            np.ndarray of shape (n_samples,): The predicted class labels

        """
        return self.classifier.predict(X)