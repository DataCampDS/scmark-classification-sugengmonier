import numpy as np
import pandas as pd
import anndata
import scvi
from scvi.model import SCVI
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import sparse

scvi.settings.verbosity = 0

class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.k_best = 800

        self.n_latent = 16 
        self.max_epochs = 400 
        self.patience = 20
        self.n_hidden = 64
        self.n_layers = 2
        self.dropout_rate = 0.3

        self.max_iter = 1000      
        self.learning_rate = 0.01 
        self.max_depth = 4        
        self.min_samples_leaf = 3 
        self.l2_regularization = 2.0 

        self.le = LabelEncoder()
        self.vae = None
        self.clf = None 
        self.selector = None
        self.selected_indices = None
        self.classes_ = None

    def fit(self, X, y):
        print("Supervised Feature Selection")

        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_

        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        X_log = np.log1p(X_dense)

        self.selector = SelectKBest(f_classif, k=self.k_best)
        self.selector.fit(X_log, y_encoded)

        mask = self.selector.get_support()
        self.selected_indices = np.where(mask)[0]

        X_filtered = X[:, self.selected_indices]

        adata_train = anndata.AnnData(X=X_filtered)
        adata_train.obs['cell_type'] = y_encoded
        adata_train.obs['batch'] = 'train_batch'

        SCVI.setup_anndata(adata_train, batch_key='batch', labels_key='cell_type')

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
            'enable_progress_bar': False
        }
        self.vae.train(**trainer_kwargs)

        X_train_vae = self.vae.get_latent_representation(adata_train)

        self.clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            scoring='loss',
            early_stopping=True,
            random_state=42
        )

        self.clf.fit(X_train_vae, y_encoded)

        return self

    def predict_proba(self, X):
        if self.selector is None:
            raise RuntimeError("Model not fitted yet!")

        X_filtered = X[:, self.selected_indices]

        adata_test = anndata.AnnData(X=X_filtered)
        adata_test.obs['batch'] = 'train_batch'
        n_obs = adata_test.n_obs
        adata_test.obs['cell_type'] = np.zeros(n_obs, dtype=int)

        try:
            SCVI.setup_anndata(adata_test, batch_key='batch', labels_key='cell_type')
        except:
            pass

        X_test_vae = self.vae.get_latent_representation(adata_test)

        return self.clf.predict_proba(X_test_vae)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(proba, axis=1))
