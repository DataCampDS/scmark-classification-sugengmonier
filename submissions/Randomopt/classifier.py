import numpy as np
import scanpy as sc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Classifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier with integrated Scanpy preprocessing and PCA pipeline.
    """

    def __init__(self):
        # State variables
        self.le_ = None             # LabelEncoder
        self.scaler_ = None         # StandardScaler
        self.pca_ = None            # PCA transformer
        self.gene_mask_ = None      # Boolean mask for HVG selection
        self.model_ = None          # RandomForestClassifier instance

    def _build_rf_model(self):
        """
        Instantiate the Random Forest model with optimized hyperparameters.
        """
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            max_features=0.4,
            min_samples_leaf=7,
            class_weight="balanced",
            max_samples=0.7,
            n_jobs=-1,
            random_state=42
        )

    def _preprocess(self, X, fit=False):
        """
        Full preprocessing pipeline:
        1. Convert to AnnData
        2. Normalize & Log1p
        3. Feature Selection (Top 2000 HVGs)
        4. Standard Scaling
        5. PCA Reduction (80 components)
        """
        # 1. Convert to Scanpy AnnData object
        adata = sc.AnnData(X)

        # 2. Normalization (CPM) and Log Transformation
        # Note: We skip 'min_cells' filtering here to maintain column alignment for test data.
        # HVG selection effectively filters out noise.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # 3. Feature Selection (Highly Variable Genes)
        if fit:
            # Training phase: Identify top 2000 variable genes
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
            self.gene_mask_ = adata.var['highly_variable'].values
        
        # Ensure gene_mask_ exists
        if self.gene_mask_ is None:
            raise RuntimeError("Must fit before transforming (gene mask missing).")
        
        # Apply gene filtering
        adata_subset = adata[:, self.gene_mask_].copy()

        # 4. Standardization (StandardScaler) -> Convert to Dense
        X_dense = adata_subset.X
        if hasattr(X_dense, "toarray"):
            X_dense = X_dense.toarray()

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_dense)
        else:
            X_scaled = self.scaler_.transform(X_dense)

        # 5. PCA Dimensionality Reduction
        if fit:
            self.pca_ = PCA(n_components=80, random_state=42)
            X_pca = self.pca_.fit_transform(X_scaled)
        else:
            X_pca = self.pca_.transform(X_scaled)

        return X_pca

    def fit(self, X, y):
        """
        Fit the pipeline and the model to the training data.
        """
        # 1. Label Encoding (String -> Int)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)

        # 2. Run preprocessing + PCA (fit=True)
        X_final = self._preprocess(X, fit=True)

        # 3. Train Random Forest
        self.model_ = self._build_rf_model()
        self.model_.fit(X_final, y_enc)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        # 1. Preprocessing + PCA (fit=False, using stored parameters)
        X_final = self._preprocess(X, fit=False)

        # 2. Predict probabilities
        return self.model_.predict_proba(X_final)
