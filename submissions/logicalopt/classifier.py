import numpy as np
import scanpy as sc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Classifier(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with integrated Scanpy preprocessing and PCA pipeline.
    """

    def __init__(self):
        # State variables
        self.le_ = None             # LabelEncoder
        self.scaler_ = None         # StandardScaler
        self.pca_ = None            # PCA transformer
        self.gene_mask_ = None      # HVG selection mask
        self.model_ = None          # LogisticRegression instance

    def _build_model(self):
        """
        Configure Logistic Regression with your specified parameters.
        """
        return LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=500,           # Increased iterations for convergence
            n_jobs=-1,
            random_state=42
        )

    def _preprocess(self, X, fit=False):
        """
        Preprocessing pipeline:
        1. Scanpy (Normalize -> Log1p)
        2. Feature Selection (Top 2000 HVGs)
        3. Standard Scaling
        4. PCA (80 components)
        """
        # 1. Convert to AnnData
        adata = sc.AnnData(X)

        # 2. Normalize and Log Transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # 3. Feature Selection (HVG)
        if fit:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
            self.gene_mask_ = adata.var['highly_variable'].values
        
        if self.gene_mask_ is None:
            raise RuntimeError("Must fit before transforming (gene mask missing).")
        
        # Apply mask
        adata_subset = adata[:, self.gene_mask_].copy()

        # 4. Scaling (StandardScaler)
        # Convert sparse to dense for scaling/PCA
        X_dense = adata_subset.X
        if hasattr(X_dense, "toarray"):
            X_dense = X_dense.toarray()

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_dense)
        else:
            X_scaled = self.scaler_.transform(X_dense)

        # 5. PCA Reduction
        if fit:
            self.pca_ = PCA(n_components=80, random_state=42)
            X_pca = self.pca_.fit_transform(X_scaled)
        else:
            X_pca = self.pca_.transform(X_scaled)

        return X_pca

    def fit(self, X, y):
        """
        Fit preprocessing and model.
        """
        # Encode labels
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)

        # Preprocess data (fit=True)
        X_final = self._preprocess(X, fit=True)

        # Train model
        self.model_ = self._build_model()
        self.model_.fit(X_final, y_enc)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        # Preprocess data (fit=False)
        X_final = self._preprocess(X, fit=False)

        return self.model_.predict_proba(X_final)
