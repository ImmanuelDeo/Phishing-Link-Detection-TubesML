import os
import pandas as pd
import joblib
from src.utils.helpers import get_project_root

DEFAULT_MODEL_DIR = os.path.join(get_project_root(), "Artifacts", "models")

class PhishingPredictor:
    def __init__(self, model_path=None):

        if model_path is None:
            model_path = os.path.join(DEFAULT_MODEL_DIR, "best_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}. "
                "Jalankan training pipeline terlebih dahulu."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        print(f"Model dimuat dari: {model_path}")

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict_single(self, features_dict):
        X = pd.DataFrame([features_dict])
        label = self.predict(X)[0]
        proba = self.predict_proba(X)[0]
        result = {
            "label": int(label),
            "label_name": "Phishing" if label == 1 else "Legitimate",
            "probability_legitimate": float(proba[0]),
            "probability_phishing": float(proba[1]),
        }
        print(f"Prediksi: {result['label_name']} "
              f"(confidence: {max(proba):.4f})")
        return result
        
    def __repr__(self):
        return f"PhishingPredictor(model_path='{self.model_path}')"