import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CORRELATION_THRESHOLD = 0.90
DEFAULT_K_BEST = 21


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
        self.tld_freq_map = None
        self.high_corr_features = None
        self.selector = None
        self.selected_features = None

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

    def encode_tld(self):
        if self.X_train is None:
            raise ValueError("Jalankan split_data terlebih dahulu.")

        if "TLD" not in self.X_train.columns:
            print("Kolom 'TLD' tidak ditemukan, skip frequency encoding.")
            return self.X_train, self.X_test

        self.tld_freq_map = self.X_train["TLD"].value_counts(normalize=True)
        self.X_train["TLD"] = self.X_train["TLD"].map(self.tld_freq_map)
        self.X_test["TLD"] = self.X_test["TLD"].map(self.tld_freq_map)
        self.X_test["TLD"] = self.X_test["TLD"].fillna(0)

        print("Frequency encoding TLD selesai.")
        return self.X_train, self.X_test

    def remove_high_correlation(self):
        if self.X_train is None:
            raise ValueError("Jalankan split_data terlebih dahulu.")

        corr_matrix = self.X_train.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.high_corr_features = [
            col for col in upper.columns
            if any(upper[col] > self.correlation_threshold)
        ]

        self.X_train = self.X_train.drop(columns=self.high_corr_features, errors="ignore")
        self.X_test = self.X_test.drop(columns=self.high_corr_features, errors="ignore")

        print(f"Fitur berkorelasi tinggi (>{self.correlation_threshold}): "
            f"{self.high_corr_features}")
        print(f"Shape setelah drop: Train {self.X_train.shape}, Test {self.X_test.shape}")

        return self.X_train, self.X_test

    def select_features(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Jalankan split_data terlebih dahulu.")

        self.selector = SelectKBest(score_func=f_classif, k=self.k_best)
        self.selector.fit(self.X_train, self.y_train)

        self.selected_features = self.X_train.columns[
            self.selector.get_support()
        ].tolist()

        self.X_train = pd.DataFrame(
            self.selector.transform(self.X_train),
            columns=self.selected_features,
            index=self.X_train.index
        )

        self.X_test = pd.DataFrame(
            self.selector.transform(self.X_test),
            columns=self.selected_features,
            index=self.X_test.index
        )

        print(f"Selected {self.k_best} features: {self.selected_features}")
        print(f"Shape setelah seleksi: Train {self.X_train.shape}, Test {self.X_test.shape}")

        return self.X_train, self.X_test

    def run_all(self, X, y):
        self.split_data(X, y)
        self.encode_tld()
        self.remove_high_correlation()
        self.select_features()

        print("\nPreprocessing selesai.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_selected_features(self):
        if self.selected_features is None:
            raise ValueError("Jalankan select_features terlebih dahulu.")
        return self.selected_features

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
