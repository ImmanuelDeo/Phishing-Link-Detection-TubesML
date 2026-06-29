import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from src.utils.helpers import ensure_dir, get_project_root

DEFAULT_FIGURES_DIR = os.path.join(get_project_root(), "Artifacts", "figures")

class ModelEvaluator:
    def __init__(self, figures_dir=None):
        self.figures_dir = figures_dir or DEFAULT_FIGURES_DIR
        ensure_dir(self.figures_dir)
        plt.style.use("ggplot")
        sns.set_palette("Set2")

    def compute_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)
            metrics["PR AUC"] = average_precision_score(y_true, y_proba)
        return metrics

    def print_classification_report(self, y_true, y_pred, model_name="Model"):
        print(f"\n{'='*60}")
        print(f"Classification Report — {model_name}")
        print(f"{'='*60}")
        target_names = ["Legitimate", "Phishing"]
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report)

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"],
        )
        ax.set_title(f"Confusion Matrix: {model_name}",
                      fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()

        if save:
            filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            filepath = os.path.join(self.figures_dir, filename)
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            print(f"Confusion matrix disimpan ke: {filepath}")
        plt.close()

    def plot_all_confusion_matrices(self, y_true, predictions, save=True):
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_true, pred["y_pred"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"],
            )
            axes[idx].set_title(f"Confusion Matrix: {model_name}",
                                fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("True Label")
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.figures_dir, "confusion_matrices_all.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            print(f"Confusion matrices disimpan ke: {filepath}")
        plt.close()

    def plot_roc_curve(self, y_true, predictions, results_df, save=True):
        plt.figure(figsize=(8, 6))
        for model_name, pred in predictions.items():
            fpr, tpr, _ = roc_curve(y_true, pred["y_proba"])
            auc_score = results_df.loc[model_name, "Test ROC AUC"]
            plt.plot(fpr, tpr,
                     label=f"{model_name} (AUC = {auc_score:.4f})",
                     linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.figures_dir, "roc_curve_comparison.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            print(f"ROC curve disimpan ke: {filepath}")
        plt.close()

    def plot_model_comparison(self, results_df, save=True):
        metrics_to_plot = [
            "Test Accuracy", "Test Precision", "Test Recall",
            "Test F1-Score", "Test ROC AUC", "Test PR AUC",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(
                data=results_df.reset_index(),
                x="Model",
                y=metric,
                ax=axes[i],
                palette="muted",
            )
            axes[i].set_title(metric, fontsize=14, fontweight="bold")
            axes[i].set_ylim(0.90, 1.01)
            axes[i].set_ylabel("Score")
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.figures_dir, "model_comparison.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            print(f"Model comparison disimpan ke: {filepath}")
        plt.close()

    def plot_feature_importance(self, pipeline, model_name, top_n=15, save=True):
        corr_features = pipeline.named_steps["corr_filter"].keep_cols_
        anova_support = pipeline.named_steps["anova"].get_support()
        selected_features = np.array(corr_features)[anova_support]
        classifier = pipeline.named_steps["classifier"]

        if not hasattr(classifier, "feature_importances_"):
            print(f"Model {model_name} tidak memiliki feature_importances_. Skip.")
            return

        importances = classifier.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": importances,
        })
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=top_features,
            x="Importance",
            y="Feature",
            palette="viridis",
        )
        plt.title(f"Top {top_n} Most Important Features — {model_name}",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()

        if save:
            filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
            filepath = os.path.join(self.figures_dir, filename)
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            print(f"Feature importance disimpan ke: {filepath}")

        plt.close()
        return importance_df