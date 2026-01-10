import numpy as np
import xgboost as xgb
import scanpy as sc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

class Classifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost Classifier with Class Weights and integrated Scanpy preprocessing.
    """

    def __init__(self):
        # State variables
        self.le_ = None             # LabelEncoder
        self.scaler_ = None         # StandardScaler
        self.gene_mask_ = None      # Boolean mask for HVG selection
        self.model_ = None          # XGBoost model instance

    def _build_xgb_model(self, n_classes):
        """
        Configure XGBoost with specific parameters.
        """
        return xgb.XGBClassifier(
            objective="multi:softprob",  # Output probabilities
            num_class=n_classes,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=200,
            tree_method="hist",          # Fast histogram-based method
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
        # Convert sparse to dense for scaling
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
        Fit the pipeline and model to the training data.
        """
        # 1. Label Encoding (String -> Int)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        classes = np.unique(y_enc)

        # 2. Calculate Class Weights (Handling Imbalance)
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=classes, 
            y=y_enc
        )
        # Map weights to each sample
        weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([weight_dict[c] for c in y_enc])

        # 3. Run preprocessing (fit=True)
        X_processed = self._preprocess(X, fit=True)

        # 4. Train XGBoost with Sample Weights
        self.model_ = self._build_xgb_model(n_classes=len(classes))
        self.model_.fit(
            X_processed, 
            y_enc, 
            sample_weight=sample_weights # Apply weights here
        )

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        # 1. Preprocessing (fit=False)
        X_processed = self._preprocess(X, fit=False)

        # 2. Predict probabilities
        return self.model_.predict_proba(X_processed)
