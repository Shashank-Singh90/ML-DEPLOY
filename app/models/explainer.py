"""Model explainer for IoT threat detection predictions."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Provides explanations for IoT threat detection model predictions."""

    def __init__(self, model_service):
        """Initialize explainer with model service."""
        self.model_service = model_service
        self.explainer_available = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the model explainer system."""
        try:
            logger.info("Initializing prediction explainer...")
            # Use model's built-in feature importance for explanations
            # Future enhancement: integrate SHAP for local explanations
            self.explainer_available = "feature_importance_analysis"
            logger.info("Feature importance explainer ready")

        except Exception as e:
            logger.error(f"Failed to initialize explainer: {str(e)}")
            self.explainer_available = None
    
    def explain_prediction(self, input_features: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """Generate comprehensive explanation for a threat prediction."""
        try:
            if not self.explainer_available:
                return {"error": "Explainer system not available"}

            # Prepare input features for analysis
            if isinstance(input_features, dict):
                features_dataframe = pd.DataFrame([input_features])
            else:
                features_dataframe = pd.DataFrame(input_features)

            # Process features using same pipeline as prediction
            processed_features = features_dataframe[self.model_service.feature_names].fillna(0)
            scaled_features = self.model_service.feature_scaler.transform(processed_features)

            # Get model prediction and confidence
            threat_prediction = self.model_service.model.predict(scaled_features)[0]
            class_probabilities = self.model_service.model.predict_proba(scaled_features)[0]

            # Analyze global feature importance from model
            global_feature_importance = self.model_service.model.feature_importances_

            # Perform feature value analysis
            detailed_feature_analysis = self._analyze_feature_significance(
                processed_features, global_feature_importance
            )

            # Sort by combined significance score
            detailed_feature_analysis.sort(key=lambda x: x['combined_significance'], reverse=True)

            # Select top important features
            most_important_features = detailed_feature_analysis[:top_k]

            # Generate human-readable explanations
            feature_explanations = []
            for feature_info in most_important_features:
                value_description = self._describe_feature_value(feature_info)
                explanation_text = (f"{feature_info['feature_name']} = {feature_info['feature_value']:.3f} "
                                  f"({value_description}, importance: {feature_info['global_importance']:.3f})")
                feature_explanations.append(explanation_text)

            # Determine confidence level
            threat_probability = class_probabilities[1]
            model_confidence = max(class_probabilities)
            confidence_category = self._categorize_confidence(model_confidence)

            # Build comprehensive explanation
            explanation_result = {
                'explanation_method': 'Feature Importance with Value Analysis',
                'prediction_details': {
                    'threat_prediction': int(threat_prediction),
                    'threat_probability': float(threat_probability),
                    'confidence_category': confidence_category,
                    'most_important_features': most_important_features,
                    'feature_explanations': feature_explanations
                },
                'human_interpretation': self._create_human_interpretation(
                    most_important_features, threat_probability, threat_prediction
                ),
                'analyzed_features_count': len(detailed_feature_analysis),
                'model_confidence_score': f"{model_confidence:.3f}"
            }

            return explanation_result

        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return {
                'error': f'Failed to generate explanation: {str(e)}',
                'fallback_option': 'Global feature importance available via /model/features endpoint'
            }
    
    def _analyze_feature_significance(self, features_dataframe: pd.DataFrame, global_importance: np.ndarray) -> List[Dict]:
        """Analyze feature significance against typical ranges."""
        feature_significance_analysis = []

        # Load training data statistics for comparison
        training_statistics = self._load_training_statistics()

        for feature_index, (feature_name, importance) in enumerate(
            zip(self.model_service.feature_names, global_importance)
        ):
            current_feature_value = float(features_dataframe.iloc[0, feature_index])

            # Determine typical range for this feature
            if training_statistics and feature_name in training_statistics:
                feature_stats = training_statistics[feature_name]
                typical_value_range = {
                    'low_quartile': feature_stats['q25'],
                    'high_quartile': feature_stats['q75'],
                    'average': feature_stats['mean'],
                    'standard_deviation': feature_stats['std']
                }

                # Calculate how unusual this value is (z-score based)
                if feature_stats['std'] > 0:
                    z_score = abs((current_feature_value - feature_stats['mean']) / feature_stats['std'])
                    unusualness_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1 range
                else:
                    unusualness_score = 0.0
            else:
                # Default ranges when training stats unavailable
                typical_value_range = {
                    'low_quartile': 0, 'high_quartile': 1,
                    'average': 0.5, 'standard_deviation': 0.3
                }
                unusualness_score = 0.0

            # Combined significance: importance weighted by how unusual the value is
            combined_significance = float(importance) * (1 + unusualness_score)

            feature_significance_analysis.append({
                'feature_name': feature_name,
                'feature_value': current_feature_value,
                'global_importance': float(importance),
                'unusualness_score': unusualness_score,
                'combined_significance': combined_significance,
                'typical_range': typical_value_range,
                'value_category': 'unusual' if unusualness_score > 0.5 else 'typical'
            })

        return feature_significance_analysis
    
    def _load_training_statistics(self) -> Dict:
        """Load or compute training data statistics"""
        try:
            # Try to load cached statistics
            stats_path = 'models/production/feature_stats.json'
            if os.path.exists(stats_path):
                import json
                with open(stats_path, 'r') as f:
                    return json.load(f)
            
            # Compute from training data if available
            data_path = 'data/raw/synthetic_iot_data.csv'
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                X = self.model_service.prepare_features(df)
                
                stats = {}
                for col in X.columns:
                    stats[col] = {
                        'mean': float(X[col].mean()),
                        'std': float(X[col].std()),
                        'q25': float(X[col].quantile(0.25)),
                        'q75': float(X[col].quantile(0.75)),
                        'min': float(X[col].min()),
                        'max': float(X[col].max())
                    }
                
                # Cache the statistics
                os.makedirs('models/production', exist_ok=True)
                import json
                with open(stats_path, 'w') as f:
                    json.dump(stats, f)
                
                return stats
                
        except Exception as e:
            logger.warning(f"Could not load training statistics: {str(e)}")
        
        return {}
    
    def _describe_feature_value(self, feature_info: Dict) -> str:
        """Generate human-readable description of feature value."""
        if feature_info['unusualness_score'] > 0.5:
            if feature_info['feature_value'] > feature_info['typical_range']['high_quartile']:
                return "unusually high value"
            elif feature_info['feature_value'] < feature_info['typical_range']['low_quartile']:
                return "unusually low value"
            else:
                return "atypical but within normal range"
        else:
            return "typical value"

    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize model confidence level."""
        if confidence_score > 0.8:
            return "HIGH"
        elif confidence_score > 0.6:
            return "MEDIUM"
        else:
            return "LOW"

    def _create_human_interpretation(self, important_features: List[Dict], threat_probability: float, prediction: int) -> str:
        """Generate human-readable interpretation of the prediction."""
        try:
            threat_classification = "THREAT DETECTED" if prediction == 1 else "NORMAL TRAFFIC"
            confidence_description = "high" if threat_probability > 0.8 or threat_probability < 0.2 else "moderate"

            interpretation = (f"This IoT network traffic is classified as {threat_classification} "
                            f"with {confidence_description} confidence ({threat_probability:.1%} threat probability). ")

            # Identify unusual patterns in top features
            unusual_patterns = [f for f in important_features[:3] if f['value_category'] == 'unusual']
            highly_important_features = [f for f in important_features[:3] if f['global_importance'] > 0.1]

            if unusual_patterns:
                unusual_feature_names = [f['feature_name'] for f in unusual_patterns]
                interpretation += f"Unusual patterns detected in: {', '.join(unusual_feature_names)}. "

            if highly_important_features:
                key_feature_names = [f['feature_name'] for f in highly_important_features]
                interpretation += f"Key decision factors: {', '.join(key_feature_names)}."

            return interpretation

        except Exception as e:
            logger.warning(f"Failed to generate interpretation: {str(e)}")
            return f"IoT traffic classified as: {'THREAT' if prediction == 1 else 'NORMAL'} (probability: {threat_probability:.1%})"
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of model feature importance."""
        try:
            if not hasattr(self.model_service.model, 'feature_importances_'):
                return {'error': 'Model does not provide feature importance information'}

            global_importance_scores = self.model_service.model.feature_importances_

            # Build feature importance summary
            feature_importance_list = []
            for feature_name, importance_score in zip(self.model_service.feature_names, global_importance_scores):
                feature_importance_list.append({
                    'feature_name': feature_name,
                    'importance_score': float(importance_score)
                })

            # Sort by importance (highest first)
            feature_importance_list.sort(key=lambda x: x['importance_score'], reverse=True)

            return {
                'global_feature_importance': feature_importance_list,
                'top_5_most_important': feature_importance_list[:5],
                'analysis_description': 'Global feature importance from Random Forest model',
                'explanation_method': 'Random Forest built-in feature importance',
                'total_features_analyzed': len(feature_importance_list)
            }

        except Exception as e:
            logger.error(f"Failed to generate feature summary: {str(e)}")
            return {'error': str(e)}
    
    @property
    def explainer(self):
        """Compatibility property for main app."""
        return self.explainer_available

    def get_explainer_status(self) -> Dict[str, Any]:
        """Get current status of the explanation system."""
        return {
            'explainer_operational': self.explainer_available is not None,
            'explanation_type': 'Feature Importance with Value Analysis',
            'advanced_explanations_available': False,  # Future: SHAP integration
            'supported_analysis_types': [
                'global_feature_importance',
                'local_value_analysis',
                'deviation_detection'
            ],
            'future_enhancements': 'SHAP-based local explanations planned'
        }