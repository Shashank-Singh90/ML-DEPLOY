import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure the repository root is on sys.path for direct execution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.runtime import (
    EXPECTED_FEATURES,
    FEATURE_FILENAME,
    FEATURE_STATS_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMLflowTrainer:
    def __init__(self, experiment_name="iot-threat-detection"):
        mlflow.set_tracking_uri("file:./mlruns")

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {str(e)}")
            raise

    def load_and_prepare_data(self):
        logger.info("Loading IoT dataset...")

        data_path = 'data/raw/synthetic_iot_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        feature_columns = [feature for feature in EXPECTED_FEATURES if feature in df.columns]
        if not feature_columns:
            raise ValueError("Training data missing required IoT network features")

        X = df[feature_columns].copy()
        y = (df['label'] > 0).astype(int)

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Attack rate: {y.mean():.2%}")

        return X, y, df
    
    def train_model_with_tracking(self, model, model_name, X_train, y_train, X_test, y_test, params=None):
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
            if params:
                mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)

            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                mlflow.log_metric("cv_f1_mean", cv_mean)
                mlflow.log_metric("cv_f1_std", cv_std)
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")
                cv_mean = f1

            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc_roc", auc)

            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                top_features = feature_importance.head(10)
                top_features_text = ", ".join([f"{row['feature']}({row['importance']:.3f})"
                                             for _, row in top_features.iterrows()])
                mlflow.log_param("top_10_features", top_features_text)

            logger.info(f"{model_name} - F1: {f1:.4f}, AUC: {auc:.4f}, CV: {cv_mean:.4f}")

            return model, f1
    
    def save_production_model(self, model, model_name, X_train, scaler):
        logger.info(f"Saving production model: {model_name}")

        production_dir = Path('models/production')
        production_dir.mkdir(parents=True, exist_ok=True)

        with (production_dir / MODEL_FILENAME).open('wb') as handle:
            pickle.dump(model, handle)

        with (production_dir / SCALER_FILENAME).open('wb') as handle:
            pickle.dump(scaler, handle)

        with (production_dir / FEATURE_FILENAME).open('w', encoding='utf-8') as handle:
            handle.write('\n'.join(X_train.columns))

        # Save feature statistics for explainability
        feature_stats = {
            column: {
                'mean': float(X_train[column].mean()),
                'std': float(X_train[column].std(ddof=0) or 0.0),
                'q25': float(X_train[column].quantile(0.25)),
                'q75': float(X_train[column].quantile(0.75)),
            }
            for column in X_train.columns
        }
        with (production_dir / FEATURE_STATS_FILENAME).open('w', encoding='utf-8') as handle:
            json.dump(feature_stats, handle)

        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'trained_at': datetime.now().isoformat(),
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train),
        }
        with (production_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)

        logger.info("Production model saved successfully")
    
    def run_experiment(self):
        logger.info("Starting IoT Threat Detection Training")

        X, y, df = self.load_and_prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        logger.info(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

        models = {}

        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr_params = {"max_iter": 1000, "class_weight": "balanced"}
        lr_model, lr_f1 = self.train_model_with_tracking(
            lr_model, "LogisticRegression", X_train_scaled, y_train, X_test_scaled, y_test, lr_params
        )
        models['LogisticRegression'] = (lr_model, lr_f1)

        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_params = {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"}
        rf_model, rf_f1 = self.train_model_with_tracking(
            rf_model, "RandomForest", X_train_scaled, y_train, X_test_scaled, y_test, rf_params
        )
        models['RandomForest'] = (rf_model, rf_f1)

        logger.info("Training Optimized Random Forest...")
        rf2_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
        rf2_params = {"n_estimators": 200, "max_depth": 15, "class_weight": "balanced"}
        rf2_model, rf2_f1 = self.train_model_with_tracking(
            rf2_model, "RandomForest_Optimized", X_train_scaled, y_train, X_test_scaled, y_test, rf2_params
        )
        models['RandomForest_Optimized'] = (rf2_model, rf2_f1)

        best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
        best_model = models[best_model_name][0]
        best_score = models[best_model_name][1]

        logger.info(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")

        self.save_production_model(best_model, best_model_name, X_train, scaler)

        with mlflow.start_run(run_name="Experiment_Summary"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_f1_score", best_score)
            mlflow.log_param("total_models_trained", len(models))
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("feature_count", len(X.columns))
            mlflow.log_param("attack_rate", y.mean())

            for name, (_, score) in models.items():
                mlflow.log_metric(f"{name}_f1_score", score)

        return best_model, best_model_name, best_score

def main():
    try:
        trainer = SimpleMLflowTrainer()
        best_model, model_name, score = trainer.run_experiment()

        print(f"\nExperiment Complete!")
        print(f"Best Model: {model_name}")
        print(f"F1 Score: {score:.4f}")
        print(f"Model saved to: models/production/")
        print(f"View results: mlflow ui")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
