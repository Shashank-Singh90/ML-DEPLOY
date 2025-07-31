import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.class_names = ['Normal', 'Attack']
        self.load_or_train_model()

    def load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = 'models/production/iot_model.pkl'
        scaler_path = 'models/production/scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info("Loading existing model...")
            self.load_model()
        else:
            logger.info("No existing model found. Training new model...")
            self.train_model()

    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        # Create binary label (0=Normal, 1=Attack)
        if 'label' in df.columns:
            df['binary_label'] = (df['label'] > 0).astype(int)
        
        # Select numeric features for the model
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
        
        # Filter to only existing columns
        available_features = [f for f in numeric_features if f in df.columns]
        self.feature_names = available_features
        
        return df[available_features]

    def train_model(self):
        """Train the IoT threat detection model"""
        try:
            # Load training data
            data_path = 'data/raw/synthetic_iot_data.csv'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found at {data_path}")
            
            logger.info(f"Loading training data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Prepare features and target
            X = self.prepare_features(df)
            y = (df['label'] > 0).astype(int)  # Binary classification
            
            logger.info(f"Training data shape: {X.shape}")
            logger.info(f"Attack rate: {y.mean():.2%}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with class balancing
            logger.info("Training Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',  # Handle imbalanced data
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully!")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=self.class_names)}")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save_model(self):
        """Save the trained model and scaler"""
        # Ensure directory exists
        Path('models/production').mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        with open('models/production/iot_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
            
        with open('models/production/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save feature names
        with open('models/production/feature_names.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))
            
        logger.info("Model saved successfully")

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            with open('models/production/iot_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
                
            with open('models/production/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
                
            # Load feature names
            with open('models/production/feature_names.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features):
        """Make prediction with confidence scores"""
        try:
            # Convert input to DataFrame
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            else:
                df = pd.DataFrame(features)
            
            # Prepare features (use only the features the model was trained on)
            X = df[self.feature_names].fillna(0)  # Fill missing values with 0
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = {
                'prediction': int(prediction),
                'prediction_label': self.class_names[prediction],
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'attack': float(probabilities[1])
                },
                'threat_score': float(probabilities[1]),  # Probability of attack
                'top_features': [{'feature': feat, 'importance': float(imp)} for feat, imp in top_features]
            }
            
            # Add risk assessment
            threat_score = result['threat_score']
            if threat_score > 0.8:
                result['risk_level'] = 'HIGH'
                result['recommended_action'] = 'Immediate investigation required'
            elif threat_score > 0.5:
                result['risk_level'] = 'MEDIUM'  
                result['recommended_action'] = 'Monitor closely'
            else:
                result['risk_level'] = 'LOW'
                result['recommended_action'] = 'Continue normal monitoring'
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
