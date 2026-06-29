import os
import warnings
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils.helpers import get_project_root, ensure_dir

PROJECT_ROOT = get_project_root()
DEFAULT_MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "Artifacts", "models")
DEFAULT_FIGURES_DIR = os.path.join(PROJECT_ROOT, "Artifacts", "figures")

class TrainingPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.90)
        self.k_best = self.config.get("k_best", 15)
        self.cv_splits = self.config.get("cv_splits", 10)
        self.model_save_dir = self.config.get("model_save_dir", DEFAULT_MODEL_SAVE_DIR)
        self.figures_dir = self.config.get("figures_dir", DEFAULT_FIGURES_DIR)
        self.invert_label = self.config.get("invert_label", True)
        self.loader = None
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        self.results_df = None
        self.best_model_name = None
        self.best_pipeline = None

    def run(self):
        warnings.filterwarnings("ignore")
        print("PHISHING LINK DETECTION — TRAINING PIPELINE")
        print("\n[1/7] Loading dataset...")
        self.loader = DataLoader()
        df = self.loader.load_raw_data()
        self.loader.print_summary()
        print("\n[2/7] Extracting features & target...")
        X, y = self.loader.get_features_and_target(invert_label=self.invert_label)
        print(f"Target distribution:")
        print(y.value_counts(normalize=True).rename(
            {0: "Legitimate", 1: "Phishing"}
        ))
        print("\n[3/7] Splitting data...")
        self.preprocessor = DataPreprocessor(
            test_size=self.test_size,
            random_state=self.random_state,
            correlation_threshold=self.correlation_threshold,
            k_best=self.k_best,
        )
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        print("\n[4/7] Training & evaluating models...")
        self.trainer = ModelTrainer(
            cv_splits=self.cv_splits,
            random_state=self.random_state,
        )
        self.results_df = self.trainer.train_all(
            self.preprocessor, X_train, y_train, X_test, y_test
        )
        self.best_model_name = self.trainer.get_best_model_name(self.results_df)
        self.best_pipeline = self.trainer.get_best_pipeline(self.results_df)
        print(f"\nModel terbaik: {self.best_model_name}")
        print("\n[5/7] Generating evaluation visualizations...")
        self.evaluator = ModelEvaluator(figures_dir=self.figures_dir)
        for model_name, pred in self.trainer.predictions.items():
            self.evaluator.print_classification_report(
                y_test, pred["y_pred"], model_name
            )
        self.evaluator.plot_all_confusion_matrices(
            y_test, self.trainer.predictions
        )
        self.evaluator.plot_roc_curve(
            y_test, self.trainer.predictions, self.results_df
        )
        self.evaluator.plot_model_comparison(self.results_df)
        self.evaluator.plot_feature_importance(
            self.best_pipeline, self.best_model_name
        )
        print("\n[6/7] Saving best model...")
        best_model_path = os.path.join(self.model_save_dir, "best_model.pkl")
        self.trainer.save_model(self.best_pipeline, best_model_path)
        for model_name, pipeline in self.trainer.trained_pipelines.items():
            filename = f"{model_name.lower().replace(' ', '_')}_pipeline.pkl"
            model_path = os.path.join(self.model_save_dir, filename)
            self.trainer.save_model(pipeline, model_path)
        print("\n" + "=" * 60)
        print("[7/7] TRAINING PIPELINE SELESAI!")
        print("=" * 60)
        print(f"Model terbaik       : {self.best_model_name}")
        print(f"Test ROC AUC        : "
              f"{self.results_df.loc[self.best_model_name, 'Test ROC AUC']:.4f}")
        print(f"Test F1-Score       : "
              f"{self.results_df.loc[self.best_model_name, 'Test F1-Score']:.4f}")
        print(f"Test Accuracy       : "
              f"{self.results_df.loc[self.best_model_name, 'Test Accuracy']:.4f}")
        print(f"Model disimpan di   : {self.model_save_dir}")
        print(f"Visualisasi di      : {self.figures_dir}")
        print("=" * 60)
        return self.results_df