import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CORRELATION_THRESHOLD = 0.90
DEFAULT_K_BEST = 15

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols or []
        self.maps_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            if col in X.columns:
                self.maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col, mapping in self.maps_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].map(mapping).fillna(0).astype(np.float32)
        return X_out

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=DEFAULT_CORRELATION_THRESHOLD):
        self.threshold = threshold
        self.keep_cols_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        corr = X_df.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        self.keep_cols_ = [
            col for col in X_df.columns
            if not any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X_df[self.keep_cols_]

class DataPreprocessor:
    def __init__(self, test_size=DEFAULT_TEST_SIZE,
                 random_state=DEFAULT_RANDOM_STATE,
                 correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
                 k_best=DEFAULT_K_BEST):
        self.test_size = test_size
        self.random_state = random_state
        self.correlation_threshold = correlation_threshold
        self.k_best = k_best
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Training target distribution:\n{self.y_train.value_counts(normalize=True)}")
        print(f"Test target distribution:\n{self.y_test.value_counts(normalize=True)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_pipeline(self, classifier):
        k_best = self.k_best
        if self.X_train is not None and isinstance(k_best, int):
            if k_best > self.X_train.shape[1]:
                k_best = "all"
        categorical_cols = []
        if self.X_train is not None:
            categorical_cols = [
                col for col in ["TLD"]
                if col in self.X_train.columns
            ]
        pipeline = Pipeline([
            ("freq_encoder", FrequencyEncoder(cols=categorical_cols)),
            ("corr_filter", CorrelationFilter(threshold=self.correlation_threshold)),
            ("anova", SelectKBest(score_func=f_classif, k=k_best)),
            ("classifier", classifier),
        ])
        return pipeline

    def get_split_data(self):
        if self.X_train is None:
            raise ValueError("Jalankan split_data terlebih dahulu.")
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def __repr__(self):
        train_shape = self.X_train.shape if self.X_train is not None else "None"
        test_shape = self.X_test.shape if self.X_test is not None else "None"
        return (
            f"DataPreprocessor("
            f"train_shape={train_shape}, "
            f"test_shape={test_shape}, "
            f"k_best={self.k_best}, "
            f"corr_threshold={self.correlation_threshold})"
        )