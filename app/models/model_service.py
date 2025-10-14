"""Model service for IoT threat detection."""
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
    """Service for loading and using IoT threat detection models."""

    def __init__(self):
        """Initialize the model service."""
        self.model = None
        self.feature_scaler = None
        self.feature_names = None
        self.threat_classes = ['Normal', 'Attack']
        self.model_loaded = False
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        production_model_path = 'models/production/iot_model.pkl'
        production_scaler_path = 'models/production/scaler.pkl'

        if os.path.exists(production_model_path) and os.path.exists(production_scaler_path):
            logger.info("Loading existing production model...")
            self._load_production_model()
        else:
            logger.info("No production model found. Training new model...")
            self._train_new_model()

    def _prepare_training_features(self, dataframe):
        """Prepare features for model training."""
        # Create binary threat labels (0=Normal, 1=Attack)
        if 'label' in dataframe.columns:
            dataframe['binary_threat_label'] = (dataframe['label'] > 0).astype(int)

        # Define expected IoT network features
        expected_network_features = [
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

        # Filter to only available features in the dataset
        available_features = [feature for feature in expected_network_features
                            if feature in dataframe.columns]
        self.feature_names = available_features

        return dataframe[available_features]

    def _train_new_model(self):
        """Train a new IoT threat detection model."""
        try:
            # Load training dataset
            training_data_path = 'data/raw/synthetic_iot_data.csv'
            if not os.path.exists(training_data_path):
                raise FileNotFoundError(f"Training data not found: {training_data_path}")

            logger.info(f"Loading training data from {training_data_path}")
            training_dataframe = pd.read_csv(training_data_path)

            # Prepare features and targets
            feature_matrix = self._prepare_training_features(training_dataframe)
            threat_labels = (training_dataframe['label'] > 0).astype(int)  # Binary classification

            logger.info(f"Training dataset shape: {feature_matrix.shape}")
            logger.info(f"Threat detection rate in data: {threat_labels.mean():.2%}")

            # Split into training and validation sets
            X_train, X_validation, y_train, y_validation = train_test_split(
                feature_matrix, threat_labels, test_size=0.2, random_state=42, stratify=threat_labels
            )

            # Scale features for better model performance
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_validation_scaled = self.feature_scaler.transform(X_validation)

            # Train Random Forest classifier with balanced classes
            logger.info("Training Random Forest threat detection model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',  # Handle imbalanced threat data
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(X_train_scaled, y_train)
            self.model_loaded = True

            # Evaluate model performance
            validation_predictions = self.model.predict(X_validation_scaled)
            model_accuracy = accuracy_score(y_validation, validation_predictions)

            logger.info("Model training completed successfully!")
            logger.info(f"Validation accuracy: {model_accuracy:.4f}")
            logger.info(f"Classification report:\n{classification_report(y_validation, validation_predictions, target_names=self.threat_classes)}")

            # Save trained model
            self._save_trained_model()

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.model_loaded = False
            raise

    def _save_trained_model(self):
        """Save the trained model and preprocessing components."""
        # Create production directory if it doesn't exist
        production_path = Path('models/production')
        production_path.mkdir(parents=True, exist_ok=True)

        # Save trained model
        model_file_path = production_path / 'iot_model.pkl'
        with open(model_file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        # Save feature scaler
        scaler_file_path = production_path / 'scaler.pkl'
        with open(scaler_file_path, 'wb') as scaler_file:
            pickle.dump(self.feature_scaler, scaler_file)

        # Save feature names for reference
        feature_names_path = production_path / 'feature_names.txt'
        with open(feature_names_path, 'w') as features_file:
            features_file.write('\n'.join(self.feature_names))

        logger.info("Model and components saved to production directory")

    def _load_production_model(self):
        """Load the production model and preprocessing components."""
        try:
            # Load trained model
            with open('models/production/iot_model.pkl', 'rb') as model_file:
                self.model = pickle.load(model_file)

            # Load feature scaler
            with open('models/production/scaler.pkl', 'rb') as scaler_file:
                self.feature_scaler = pickle.load(scaler_file)

            # Load feature names
            with open('models/production/feature_names.txt', 'r') as features_file:
                self.feature_names = [line.strip() for line in features_file.readlines()]

            self.model_loaded = True
            logger.info("Production model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load production model: {str(e)}")
            self.model_loaded = False
            raise

    def predict(self, network_features):
        """Make threat prediction with confidence scores."""
        try:
            if not self.model_loaded:
                raise Exception("Model not loaded - cannot make predictions")

            # Convert input to standardized format
            if isinstance(network_features, dict):
                features_dataframe = pd.DataFrame([network_features])
            else:
                features_dataframe = pd.DataFrame(network_features)

            # Prepare features using only trained feature set
            prediction_features = features_dataframe[self.feature_names].fillna(0)

            # Apply feature scaling
            scaled_features = self.feature_scaler.transform(prediction_features)

            # Generate prediction and confidence scores
            threat_prediction = self.model.predict(scaled_features)[0]
            class_probabilities = self.model.predict_proba(scaled_features)[0]

            # Extract most important features for this prediction
            global_feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_important_features = sorted(global_feature_importance.items(),
                                          key=lambda x: x[1], reverse=True)[:5]

            # Build prediction result
            prediction_result = {
                'prediction': int(threat_prediction),
                'prediction_label': self.threat_classes[threat_prediction],
                'confidence': float(max(class_probabilities)),
                'class_probabilities': {
                    'normal_traffic': float(class_probabilities[0]),
                    'threat_detected': float(class_probabilities[1])
                },
                'threat_score': float(class_probabilities[1]),  # Threat probability
                'important_features': [
                    {'feature_name': feature, 'importance_score': float(importance)}
                    for feature, importance in top_important_features
                ]
            }

            # Calculate risk level and recommendations
            threat_probability = prediction_result['threat_score']
            if threat_probability > 0.8:
                prediction_result['risk_level'] = 'HIGH'
                prediction_result['recommended_action'] = 'Immediate security investigation required'
            elif threat_probability > 0.5:
                prediction_result['risk_level'] = 'MEDIUM'
                prediction_result['recommended_action'] = 'Enhanced monitoring recommended'
            else:
                prediction_result['risk_level'] = 'LOW'
                prediction_result['recommended_action'] = 'Continue standard monitoring'

            return prediction_result

        except Exception as e:
            logger.error(f"Threat prediction failed: {str(e)}")
            raise Exception(f"Threat prediction error: {str(e)}")
