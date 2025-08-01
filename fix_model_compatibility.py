#!/usr/bin/env python3
"""
Fix Model Compatibility - Deploy Best Performing Model
This script updates the production model to use RandomForest_Optimized
"""
import pickle
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_best_model():
    """Retrain and deploy the best performing model (RandomForest_Optimized)"""
    
    logger.info("🔄 Retraining RandomForest_Optimized model for production...")
    
    # Load training data
    data_path = 'data/raw/synthetic_iot_data.csv'
    if not os.path.exists(data_path):
        logger.error(f"❌ Training data not found at {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    logger.info(f"📊 Loaded {len(df)} training samples")
    
    # Prepare features (same as ModelService)
    numeric_features = [
        'flow_duration', 'Duration', 'Rate', 'Srate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
        'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
        'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
        'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
        'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP',
        'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
        'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius',
        'Covariance', 'Variance', 'Weight'
    ]
    
    # Filter to existing features
    available_features = [f for f in numeric_features if f in df.columns]
    X = df[available_features]
    y = (df['label'] > 0).astype(int)
    
    logger.info(f"🎯 Features: {len(available_features)}, Attack rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest_Optimized (best hyperparameters from MLflow)
    logger.info("🌲 Training RandomForest_Optimized...")
    model = RandomForestClassifier(
        n_estimators=200,        # Best from MLflow experiments
        max_depth=15,           # Best from MLflow experiments  
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"✅ Model trained! F1 Score: {f1:.4f}")
    logger.info(f"📈 Classification Report:\n{classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])}")
    
    # Save production model
    production_dir = Path('models/production')
    production_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(production_dir / 'iot_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler  
    with open(production_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(production_dir / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(available_features))
    
    # Update metadata
    metadata = {
        "model_name": "RandomForest_Optimized",
        "model_type": "RandomForestClassifier", 
        "trained_at": datetime.now().isoformat(),
        "feature_count": len(available_features),
        "training_samples": len(X_train),
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 15, 
            "class_weight": "balanced"
        },
        "performance": {
            "f1_score": float(f1),
            "test_samples": len(X_test)
        },
        "features": available_features
    }
    
    with open(production_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("💾 Production model updated successfully!")
    logger.info(f"📂 Saved to: {production_dir}")
    logger.info(f"🏆 Model: RandomForestClassifier with F1={f1:.4f}")
    
    return True

def verify_model_deployment():
    """Verify the new model works correctly"""
    logger.info("🔍 Verifying model deployment...")
    
    try:
        # Import and test ModelService
        import sys
        sys.path.append('.')
        from app.models.model_service import ModelService
        
        # Initialize service (should load new model)
        service = ModelService()
        
        # Test prediction with sample data
        sample_data = {
            "flow_duration": 1.5,
            "Duration": 2.3,
            "Rate": 150.5,
            "Srate": 100.2,
            "fin_flag_number": 1,
            "syn_flag_number": 2,
            "rst_flag_number": 0,
            "psh_flag_number": 5,
            "ack_flag_number": 10,
            "ece_flag_number": 0,
            "cwr_flag_number": 0,
            "ack_count": 15,
            "syn_count": 3,
            "fin_count": 1,
            "rst_count": 0,
            "HTTP": 1,
            "HTTPS": 0,
            "DNS": 0,
            "Telnet": 0,
            "SMTP": 0,
            "SSH": 0,
            "IRC": 0,
            "TCP": 1,
            "UDP": 0,
            "DHCP": 0,
            "ARP": 0,
            "ICMP": 0,
            "IPv": 1,
            "LLC": 0,
            "Tot sum": 1500.5,
            "Min": 40.0,
            "Max": 1460.0,
            "AVG": 750.25,
            "Std": 450.3,
            "Tot size": 15000,
            "IAT": 0.01,
            "Number": 20,
            "Magnitue": 100.5,
            "Radius": 50.2,
            "Covariance": 0.8,
            "Variance": 0.64,
            "Weight": 1.0
        }
        
        result = service.predict(sample_data)
        
        logger.info("✅ Model verification successful!")
        logger.info(f"🎯 Test prediction: {result['prediction_label']}")
        logger.info(f"📊 Confidence: {result['confidence']:.3f}")
        logger.info(f"⚠️  Risk Level: {result['risk_level']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("🚀 Starting Model Compatibility Fix...")
    print("=" * 60)
    
    # Step 1: Retrain best model
    if not retrain_best_model():
        print("❌ Failed to retrain model")
        return False
    
    # Step 2: Verify deployment
    if not verify_model_deployment():
        print("❌ Failed to verify model deployment")
        return False
    
    print("=" * 60)
    print("🎉 MODEL COMPATIBILITY FIX COMPLETE!")
    print("✅ RandomForest_Optimized model deployed to production")
    print("✅ Model service verification passed")
    print("🚀 Ready to restart your API with the best model!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
