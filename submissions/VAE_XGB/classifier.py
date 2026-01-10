import numpy as np
import pandas as pd
import anndata
import scvi
from scvi.model import SCVI
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import sparse

# Keep logs clean
scvi.settings.verbosity = 0

class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):

        # --- 1. Feature Selection ---
        self.k_best = 800

        # --- 2. scVI Architecture (Lightweight) ---
        self.n_latent = 16  #As long as it's not too small, that's fine
        self.max_epochs = 400   # Increase epochs for better convergence
        self.patience = 20
        self.n_hidden = 64
        self.n_layers = 2
        self.dropout_rate = 0.3

        # --- 3. XGBoost (Best Parameters) ---
        self.n_estimators = 1000      # [Best] works well with low learning rate
        self.learning_rate = 0.01     # [Best] slower learning for higher accuracy
        self.max_depth = 4            # [Best] slightly deeper than 3 to capture subtle patterns
        self.gamma = 1.0              # [Best] main penalty term for tree growth
        self.min_child_weight = 3     # [Best] prevents leaves from being too small

        # Robustness parameters (unchanged)
        self.subsample = 0.7
        self.colsample_bytree = 0.5
        self.reg_alpha = 1.0
        self.reg_lambda = 2.0         # [Best] L2 regularization

        self.n_classes = 4

        # State variables
        self.le = LabelEncoder()
        self.vae = None
        self.xgb_clf = None
        self.selector = None
        self.selected_indices = None

    def fit(self, X, y):
        print("--- 1. Supervised Feature Selection (Best Params: k=800) ---")

        y_encoded = self.le.fit_transform(y)

        # --- Feature Selection ---
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        X_log = np.log1p(X_dense)

        self.selector = SelectKBest(f_classif, k=self.k_best)
        self.selector.fit(X_log, y_encoded)

        mask = self.selector.get_support()
        self.selected_indices = np.where(mask)[0]

        print(f"Selected top {len(self.selected_indices)} features.")

        # --- scVI Training ---
        X_filtered = X[:, self.selected_indices]

        adata_train = anndata.AnnData(X=X_filtered)
        adata_train.obs['cell_type'] = y_encoded
        adata_train.obs['batch'] = 'train_batch'

        SCVI.setup_anndata(adata_train, batch_key='batch', labels_key='cell_type')

        # print(f"--- 2. Training SCVI ---")
        self.vae = SCVI(
            adata_train,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            n_latent=self.n_latent,
            dropout_rate=self.dropout_rate
        )

        trainer_kwargs = {
            'max_epochs': self.max_epochs,
            'early_stopping_patience': self.patience,
            'check_val_every_n_epoch': 1,
            'accelerator': 'auto',
            'devices': 'auto',
            'enable_progress_bar': False # Recommended to disable during submission
        }
        self.vae.train(**trainer_kwargs)

        # --- XGBoost Training ---
        # print("--- 3. XGBoost Training (Best Params) ---")
        X_train_vae = self.vae.get_latent_representation(adata_train)

        self.xgb_clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            gamma=self.gamma,                 # Key parameter
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective='multi:softprob',
            num_class=self.n_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        self.xgb_clf.fit(X_train_vae, y_encoded)

        print("Training Complete.")
        return self

    def predict_proba(self, X):
        if self.selector is None:
            raise RuntimeError("Model not fitted yet!")

        # 1. Apply feature selection
        X_filtered = X[:, self.selected_indices]

        # 2. Construct test AnnData
        adata_test = anndata.AnnData(X=X_filtered)
        adata_test.obs['batch'] = 'train_batch'
        n_obs = adata_test.n_obs
        adata_test.obs['cell_type'] = np.zeros(n_obs, dtype=int)

        # 3. scVI Setup (must do for new data)
        try:
             SCVI.setup_anndata(adata_test, batch_key='batch', labels_key='cell_type')
        except:
             pass

        # 4. Get latent representation
        X_test_vae = self.vae.get_latent_representation(adata_test)

        # 5. Prediction
        return self.xgb_clf.predict_proba(X_test_vae)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(proba, axis=1))
