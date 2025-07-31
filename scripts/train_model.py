import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import mlflow
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
import os
import sys

# Add the app directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.models.model_service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMLflowTrainer:
    def __init__(self, experiment_name="iot-threat-detection"):
        # Set MLflow tracking URI (local for now)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment
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
        """Load and prepare the IoT dataset"""
        logger.info("Loading IoT dataset...")
        
        data_path = 'data/raw/synthetic_iot_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Prepare features using ModelService logic
        model_service = ModelService.__new__(ModelService)  # Create instance without __init__
        X = model_service.prepare_features(df)
        y = (df['label'] > 0).astype(int)  # Binary classification
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Attack rate: {y.mean():.2%}")
        
        return X, y, df
    
    def train_model_with_tracking(self, model, model_name, X_train, y_train, X_test, y_test, params=None):
        """Train a model with MLflow tracking (without model logging to avoid compatibility issues)"""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
            
            # Log parameters
            if params:
                mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')  # Reduced CV folds
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                mlflow.log_metric("cv_f1_mean", cv_mean)
                mlflow.log_metric("cv_f1_std", cv_std)
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")
                cv_mean = f1  # Fallback
            
            # Log metrics
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc_roc", auc)
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save top 10 features as text (avoiding file artifacts for now)
                top_features = feature_importance.head(10)
                top_features_text = ", ".join([f"{row['feature']}({row['importance']:.3f})" 
                                             for _, row in top_features.iterrows()])
                mlflow.log_param("top_10_features", top_features_text)
            
            logger.info(f"{model_name} - F1: {f1:.4f}, AUC: {auc:.4f}, CV: {cv_mean:.4f}")
            
            return model, f1
    
    def save_production_model(self, model, model_name, X_train, scaler):
        """Save the best model for production"""
        logger.info(f"Saving production model: {model_name}")
        
        # Ensure directory exists
        Path('models/production').mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = 'models/production/iot_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = 'models/production/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open('models/production/feature_names.txt', 'w') as f:
            f.write('\n'.join(X_train.columns))
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'trained_at': datetime.now().isoformat(),
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        with open('models/production/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Production model saved successfully")
    
    def run_experiment(self):
        """Run complete ML experiment with multiple algorithms"""
        logger.info("🚀 Starting Simple MLflow IoT Threat Detection Experiment")
        
        # Load data
        X, y, df = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        logger.info(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        
        # Train models
        models = {}
        
        # 1. Baseline - Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr_params = {"max_iter": 1000, "class_weight": "balanced"}
        lr_model, lr_f1 = self.train_model_with_tracking(
            lr_model, "LogisticRegression", X_train_scaled, y_train, X_test_scaled, y_test, lr_params
        )
        models['LogisticRegression'] = (lr_model, lr_f1)
        
        # 2. Simple Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_params = {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"}
        rf_model, rf_f1 = self.train_model_with_tracking(
            rf_model, "RandomForest", X_train_scaled, y_train, X_test_scaled, y_test, rf_params
        )
        models['RandomForest'] = (rf_model, rf_f1)
        
        # 3. Optimized Random Forest
        logger.info("Training Optimized Random Forest...")
        rf2_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
        rf2_params = {"n_estimators": 200, "max_depth": 15, "class_weight": "balanced"}
        rf2_model, rf2_f1 = self.train_model_with_tracking(
            rf2_model, "RandomForest_Optimized", X_train_scaled, y_train, X_test_scaled, y_test, rf2_params
        )
        models['RandomForest_Optimized'] = (rf2_model, rf2_f1)
        
        # Find best model
        best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
        best_model = models[best_model_name][0]
        best_score = models[best_model_name][1]
        
        logger.info(f"🏆 Best model: {best_model_name} with F1 score: {best_score:.4f}")
        
        # Save best model for production
        self.save_production_model(best_model, best_model_name, X_train, scaler)
        
        # Log experiment summary
        with mlflow.start_run(run_name="Experiment_Summary"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_f1_score", best_score)
            mlflow.log_param("total_models_trained", len(models))
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("feature_count", len(X.columns))
            mlflow.log_param("attack_rate", y.mean())
            
            # Log all model scores for comparison
            for name, (_, score) in models.items():
                mlflow.log_metric(f"{name}_f1_score", score)
        
        return best_model, best_model_name, best_score

def main():
    """Main function to run the experiment"""
    try:
        trainer = SimpleMLflowTrainer()
        best_model, model_name, score = trainer.run_experiment()
        
        print(f"\n🎉 Experiment Complete!")
        print(f"🏆 Best Model: {model_name}")
        print(f"📊 Best F1 Score: {score:.4f}")
        print(f"💾 Model saved to: models/production/")
        print(f"📈 View results: Run 'mlflow ui' in terminal or use start_mlflow.py")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
