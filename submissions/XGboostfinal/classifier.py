import numpy as np
import xgboost as xgb
import scanpy as sc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

class Classifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost Classifier with Enhanced Regularization to prevent Train=1.0.
    """

    def __init__(self):
        # State variables for preprocessing and modeling
        self.le_ = None
        self.scaler_ = None
        self.gene_mask_ = None
        self.model_ = None

    def _build_xgb_model(self, n_classes):
        """
        Configure XGBoost with stricter constraints to solve Overfitting (Train=1.0).
        """
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            
            # --- 1. 降低模型复杂度 (Capacity Control) ---
            # 降低学习率，增加树的数量（慢工出细活）
            learning_rate=0.05,      # 原来是 0.1 (降速)
            n_estimators=600,        # 原来是 500 (因为降速了，稍微加一点树)
            max_depth=4,             # 原来是 5 (变浅，防止模型记忆太复杂的细节)
            
            # --- 2. 结构性约束 (Structural Constraints) ---
            # 这是一个关键点：强制叶子节点必须包含更多样本
            min_child_weight=6,      # 原来是 3 (大幅提升，强迫模型不关注孤立样本)
            gamma=1.5,               # 原来是 0.5 (大幅提升，只有分裂增益很大时才允许分裂)
            
            # --- 3. 正则化与随机性 (Regularization & Randomness) ---
            reg_lambda=2.0,          # 原来是 1.0 (增加 L2 正则，限制权重过大)
            reg_alpha=0.5,           # 原来是 0.0 (增加 L1 正则，这对稀疏的基因数据非常有效！)
            
            subsample=0.7,           # 原来是 0.8 (减少每次看到的样本，增加随机性)
            colsample_bytree=0.6,    # 原来是 0.8 (减少每次看到的特征，防止依赖特定基因)
            
            # System parameters
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
        )

    def _preprocess(self, X, fit=False):
        """
        Preprocessing pipeline:
        1. Convert to AnnData
        2. Normalize (CPM) & Log1p
        3. Feature Selection (Top 2000 HVGs)
        4. Standard Scaling
        """
        # 1. Convert to Scanpy AnnData
        adata = sc.AnnData(X)

        # 2. Normalize and Log Transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # 3. Feature Selection (Highly Variable Genes)
        if fit:
            # Training: Identify top 2000 genes
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
            self.gene_mask_ = adata.var['highly_variable'].values
        
        # Ensure mask exists
        if self.gene_mask_ is None:
            raise RuntimeError("Must fit before transforming (gene mask missing).")
        
        # Apply mask
        adata_subset = adata[:, self.gene_mask_].copy()

        # 4. Standardization (StandardScaler)
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
        Fit the pipeline: Encode -> Weight -> Preprocess -> Train.
        """
        # 1. Label Encoding
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        classes = np.unique(y_enc)

        # 2. Calculate Class Weights (Balanced)
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=classes, 
            y=y_enc
        )
        weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([weight_dict[c] for c in y_enc])

        # 3. Preprocess Data (fit=True)
        X_processed = self._preprocess(X, fit=True)

        # 4. Train Model
        self.model_ = self._build_xgb_model(n_classes=len(classes))
        self.model_.fit(
            X_processed, 
            y_enc, 
            sample_weight=sample_weights
        )

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """
        if self.model_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        # 1. Preprocess Data (fit=False)
        X_processed = self._preprocess(X, fit=False)

        # 2. Predict
        return self.model_.predict_proba(X_processed)