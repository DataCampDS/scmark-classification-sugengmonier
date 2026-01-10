import numpy as np
import xgboost as xgb
import scanpy as sc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Classifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost Classifier (No Class Weights) with integrated Scanpy preprocessing.
    """

    def __init__(self):
        # State variables
        self.le_ = None             # LabelEncoder
        self.scaler_ = None         # StandardScaler
        self.gene_mask_ = None      # Boolean mask for HVG selection
        self.model_ = None          # XGBoost model instance

    def _build_xgb_model(self, n_classes):
        """
        Configure XGBoost with the specific parameters.
        """
        return xgb.XGBClassifier(
            objective="multi:softprob",  # Output probabilities
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

    def _preprocess(self, X, fit=False):
        """
        Preprocessing pipeline:
        1. Convert to AnnData
        2. Normalize & Log1p
        3. Feature Selection (Top 2000 HVGs)
        4. Standard Scaling
        """
        # 1. Convert to Scanpy AnnData object
        adata = sc.AnnData(X)

        # 2. Normalization (CPM) and Log Transformation
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

        # 4. Standardization (StandardScaler)
        # Convert sparse to dense for scaling and model input
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
        """
        Fit the preprocessing pipeline and the model to the training data.
        """
        # 1. Label Encoding (String -> Int)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        classes = np.unique(y_enc)

        # 2. Run preprocessing (fit=True)
        # This calculates HVGs and scaling parameters
        X_processed = self._preprocess(X, fit=True)

        # 3. Train XGBoost
        # Note: No 'sample_weight' is passed here. 
        # The model treats all classes equally.
        self.model_ = self._build_xgb_model(n_classes=len(classes))
        self.model_.fit(X_processed, y_enc)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        # 1. Preprocessing (fit=False)
        # Uses the HVGs and Scaling parameters from training
        X_processed = self._preprocess(X, fit=False)

        # 2. Predict probabilities
        return self.model_.predict_proba(X_processed)
