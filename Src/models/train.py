import os
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    make_scorer,
)
from src.utils.helpers import ensure_dir

DEFAULT_MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    ),
}

DEFAULT_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

class ModelTrainer:
    def __init__(self, models=None, scoring=None, cv_splits=10, random_state=42):
        self.models = models or DEFAULT_MODELS
        self.scoring = scoring or DEFAULT_SCORING
        self.cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state,
        )
        self.results = []
        self.predictions = {}
        self.trained_pipelines = {}
        self.cv_raw_scores = {}

    def train_single(self, name, pipeline, X_train, y_train, X_test, y_test):
        print(f"\nTraining {name}...")
        print("-" * 40)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=self.cv,
            scoring=self.scoring,
            return_train_score=False,
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "Model": name,
            "CV Accuracy": cv_result["test_accuracy"].mean(),
            "CV Precision": cv_result["test_precision"].mean(),
            "CV Recall": cv_result["test_recall"].mean(),
            "CV F1-Score": cv_result["test_f1"].mean(),
            "CV ROC AUC": cv_result["test_roc_auc"].mean(),
            "CV PR AUC": cv_result["test_pr_auc"].mean(),
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Precision": precision_score(y_test, y_pred, zero_division=0),
            "Test Recall": recall_score(y_test, y_pred, zero_division=0),
            "Test F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Test ROC AUC": roc_auc_score(y_test, y_proba),
            "Test PR AUC": average_precision_score(y_test, y_proba),
        }
        print(f"  CV Accuracy : {metrics['CV Accuracy']:.4f}")
        print(f"  CV F1-Score : {metrics['CV F1-Score']:.4f}")
        print(f"  CV ROC AUC  : {metrics['CV ROC AUC']:.4f}")
        print(f"  Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"  Test F1-Score: {metrics['Test F1-Score']:.4f}")
        print(f"  Test ROC AUC : {metrics['Test ROC AUC']:.4f}")
        return metrics, y_pred, y_proba, pipeline, cv_result

    def train_all(self, preprocessor, X_train, y_train, X_test, y_test):
        print("=" * 60)
        print("TRAINING SEMUA MODEL")
        print("=" * 60)
        self.results = []
        self.predictions = {}
        self.trained_pipelines = {}
        self.cv_raw_scores = {}
        for name, classifier in self.models.items():
            pipeline = preprocessor.build_pipeline(classifier)
            metrics, y_pred, y_proba, fitted_pipe, cv_result = self.train_single(
                name, pipeline, X_train, y_train, X_test, y_test
            )
            self.results.append(metrics)
            self.predictions[name] = {
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
            self.trained_pipelines[name] = fitted_pipe
            self.cv_raw_scores[name] = cv_result
        results_df = pd.DataFrame(self.results)
        results_df = results_df.set_index("Model")
        results_df = results_df.sort_values("Test ROC AUC", ascending=False)
        print("\n" + "=" * 60)
        print("HASIL PERBANDINGAN MODEL")
        print("=" * 60)
        print(results_df.to_string())
        return results_df

    def get_best_model_name(self, results_df):
        return results_df.index[0]

    def get_best_pipeline(self, results_df):
        best_name = self.get_best_model_name(results_df)
        return self.trained_pipelines[best_name]
    @staticmethod

    def save_model(pipeline, filepath):
        ensure_dir(os.path.dirname(filepath))
        joblib.dump(pipeline, filepath)
        print(f"Model disimpan ke: {filepath}")
        return filepath
    @staticmethod
    
    def load_model(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model tidak ditemukan: {filepath}")
        pipeline = joblib.load(filepath)
        print(f"Model dimuat dari: {filepath}")
        return pipeline