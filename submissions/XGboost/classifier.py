import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

class Classifier(object):
    """XGBoost classifier for the scMARK RAMP (multiclass)."""

    def __init__(self):
        self.le_ = None          # LabelEncoder instance
        self.model_ = None       # xgb.XGBClassifier instance

    def _build_xgb_model(self, n_classes):
        """
        Configure and return the XGBoost classifier.
        """
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
        )

    def fit(self, X, y):
        """
        Fit the model on training data.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Input feature matrix.
        y : numpy.ndarray
            Target labels (strings).
        """
        # 1. Encode string labels to integers
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)  # 0, 1, 2, ...

        n_classes = len(self.le_.classes_)

        # 2. Build and train the XGBoost model
        # Note: No class weights or sample weights are used here.
        self.model_ = self._build_xgb_model(n_classes=n_classes)
        
        self.model_.fit(
            X, y_enc
        )

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Returns
        -------
        proba : numpy.ndarray
            Shape (n_samples, n_classes). The column order matches self.le_.classes_.
        """
        if self.model_ is None or self.le_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        # XGBoost outputs probabilities in the order of the encoded integers (0..K-1),
        # which corresponds exactly to the order of self.le_.classes_.
        return self.model_.predict_proba(X)
