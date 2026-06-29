import os
import pandas as pd
from src.Features.feature_config import (
    TARGET_COLUMN,
    COLUMNS_TO_DROP,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "Data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "Data", "processed")
DEFAULT_RAW_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_PROCESSED_FILENAME = "processed_dataset.csv"

class DataLoader:
    def __init__(self, raw_dir=None, processed_dir=None,
                 raw_filename=None, processed_filename=None):
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.raw_filename = raw_filename or DEFAULT_RAW_FILENAME
        self.processed_filename = processed_filename or DEFAULT_PROCESSED_FILENAME
        self.dataframe = None

    def load_raw_data(self):
        filepath = os.path.join(self.raw_dir, self.raw_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"File dataset mentah tidak ditemukan: {filepath}"
            )
        self.dataframe = pd.read_csv(filepath)
        print(f"Dataset mentah berhasil dimuat — shape: {self.dataframe.shape}")
        return self.dataframe

    def load_processed_data(self):
        filepath = os.path.join(self.processed_dir, self.processed_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"File dataset processed tidak ditemukan: {filepath}. "
                "Jalankan preprocessing terlebih dahulu."
            )
        self.dataframe = pd.read_csv(filepath)
        print(f"Dataset processed berhasil dimuat — shape: {self.dataframe.shape}")
        return self.dataframe

    def save_processed_data(self, dataframe=None, filename=None):
        df_to_save = dataframe if dataframe is not None else self.dataframe
        if df_to_save is None:
            raise ValueError(
                "Tidak ada DataFrame untuk disimpan. "
                "Muat data terlebih dahulu atau berikan parameter dataframe."
            )
        os.makedirs(self.processed_dir, exist_ok=True)
        save_path = os.path.join(self.processed_dir, filename or self.processed_filename)
        df_to_save.to_csv(save_path, index=False)
        print(f"Dataset processed disimpan ke: {save_path}")
        return save_path

    def get_features_and_target(self, dataframe=None, target_column=TARGET_COLUMN,
                                drop_columns=None, invert_label=True):
        df = dataframe if dataframe is not None else self.dataframe
        if df is None:
            raise ValueError("Tidak ada DataFrame. Muat data terlebih dahulu.")
        cols_to_drop = drop_columns if drop_columns is not None else COLUMNS_TO_DROP
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df_model = df.drop(columns=existing_cols_to_drop)
        if target_column not in df_model.columns:
            raise ValueError(
                f"Kolom target '{target_column}' tidak ditemukan dalam DataFrame."
            )
        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]
        if invert_label:
            y = (y == 0).astype(int)
            print("Label diinversi: 1 = Phishing, 0 = Legitimate")
        print(f"Features shape: {X.shape} | Target shape: {y.shape} | "
              f"Kolom di-drop: {len(existing_cols_to_drop)}")
        return X, y

    def print_summary(self):
        if self.dataframe is None:
            print("Belum ada dataset yang dimuat.")
            return
        df = self.dataframe
        print("RINGKASAN DATASET")
        print(f"Shape              : {df.shape}")
        print(f"Jumlah fitur       : {df.shape[1]}")
        print(f"Jumlah sampel      : {df.shape[0]}")
        print(f"Memory usage       : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        total_missing = df.isnull().sum().sum()
        print(f"Total missing values: {total_missing}")
        if total_missing > 0:
            missing_per_col = df.isnull().sum()
            cols_with_missing = missing_per_col[missing_per_col > 0]
            print("Kolom dengan missing values:")
            for col, count in cols_with_missing.items():
                print(f"  - {col}: {count}")
        dup_count = df.duplicated().sum()
        print(f"Jumlah baris duplikat: {dup_count}")
        print("Tipe data:")
        for dtype, count in df.dtypes.value_counts().items():
            print(f"  - {dtype}: {count} kolom")
        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            print(f"Kolom non-numeric ({len(non_numeric)}): {non_numeric}")
        if TARGET_COLUMN in df.columns:
            print(f"Distribusi target ('{TARGET_COLUMN}'):")
            dist = df[TARGET_COLUMN].value_counts(normalize=True)
            for label, proportion in dist.items():
                label_name = "Legitimate" if label == 1 else "Phishing"
                print(f"  - {label} ({label_name}): {proportion:.4%}")
        print("=" * 60)

    def get_column_info(self):
        if self.dataframe is None:
            raise ValueError("Tidak ada DataFrame. Muat data terlebih dahulu.")
        df = self.dataframe
        info = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.values,
            "non_null_count": df.notnull().sum().values,
            "null_count": df.isnull().sum().values,
            "unique_count": df.nunique().values,
        })
        return info

    def __repr__(self):
        shape_info = self.dataframe.shape if self.dataframe is not None else "None"
        raw_path = os.path.join(self.raw_dir, self.raw_filename)
        processed_path = os.path.join(self.processed_dir, self.processed_filename)
        return (
            f"DataLoader("
            f"raw='{raw_path}', "
            f"processed='{processed_path}', "
            f"loaded_shape={shape_info})"
        )