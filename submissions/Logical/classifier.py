# classifier.py  —— Logistic Regression + PCA 版
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Classifier(object):
    """Logistic Regression with PCA for the scMARK RAMP."""

    def __init__(self):
        # 标签编码器（把字符串标签映射到 0,1,2,3）
        self.le_ = None
        # 具体的模型（一个 sklearn Pipeline）
        self.model_ = None

    def _build_pipeline(self):
        """构建和你 notebook 里一致的 logit_pipe，只是加了 to_dense。"""
        to_dense = FunctionTransformer(
            lambda X: X.toarray() if hasattr(X, "toarray") else X,
            accept_sparse=True
        )

        pipe = Pipeline(
            [
                ("to_dense", to_dense),
                ("Scaler", StandardScaler(with_mean=True, with_std=True)),
                ("PCA", PCA(n_components=80, random_state=42)),
                (
                    "LogReg",
                    LogisticRegression(
                        multi_class="multinomial",
                        class_weight="balanced",
                        max_iter=500,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        return pipe

    def fit(self, X, y):
        """
        X: 稀疏 CSR 计数矩阵（problem.get_train_data() 给的 X_train）
        y: 字符串标签数组，例如 ['Cancer_cells', 'T_cells_CD4+', ...]
        """
        # 1) 把字符串标签编码成 0,1,2,3
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)

        # 2) 构建 pipeline 并拟合（内部会做：to_dense -> 标准化 -> PCA -> LogReg）
        self.model_ = self._build_pipeline()
        self.model_.fit(X, y_enc)

        return self

    def predict_proba(self, X):
        """
        返回 shape = (n_samples, n_classes) 的概率矩阵，
        列顺序和 self.le_.classes_ 一致。
        """
        if self.model_ is None or self.le_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        proba_enc = self.model_.predict_proba(X)  # 对应编码后的 0,1,2,3

        # 已经按 0..K-1 顺序，和 self.le_.classes_ 一致，直接返回即可
        return proba_enc
