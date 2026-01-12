import numpy as np
import scanpy as sc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import HistGradientBoostingClassifier

class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.le_ = None
        self.scaler_ = None
        self.gene_mask_ = None
        self.model_ = None
        self.classes_ = None

    def _preprocess(self, X, fit=False):
        adata = sc.AnnData(X)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        if fit:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
            self.gene_mask_ = adata.var['highly_variable'].values
        
        if self.gene_mask_ is None:
            if hasattr(adata.X, "toarray"):
                return adata.X.toarray()
            return adata.X

        adata_subset = adata[:, self.gene_mask_].copy()

        X_dense = adata_subset.X
        if hasattr(X_dense, "toarray"):
            X_dense = X_dense.toarray()

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_dense)
        else:
            X_scaled = self.scaler_.transform(X_dense)

        return X_scaled

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        classes_unique = np.unique(y_enc)
        weights = compute_class_weight('balanced', classes=classes_unique, y=y_enc)
        weight_dict = dict(zip(classes_unique, weights))
        sample_weights = np.array([weight_dict[c] for c in y_enc])

        X_processed = self._preprocess(X, fit=True)

        self.model_ = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=300,
            max_depth=5,
            l2_regularization=1.0,
            early_stopping=True,
            scoring='loss',
            random_state=42
        )
        
        self.model_.fit(X_processed, y_enc, sample_weight=sample_weights)
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Fit first!")
            
        X_processed = self._preprocess(X, fit=False)
        return self.model_.predict_proba(X_processed)
