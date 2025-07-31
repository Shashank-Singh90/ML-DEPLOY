import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model_service):
        self.model_service = model_service
        self.explainer = None
        self.setup_explainer()
    
    def setup_explainer(self):
        """Initialize simplified explainer (without SHAP for now)"""
        try:
            logger.info("Setting up model explainer...")
            # For now, we'll use model's built-in feature importance
            # We can add SHAP later once we resolve the compatibility issue
            self.explainer = "feature_importance"
            logger.info("Feature importance explainer initialized")
                
        except Exception as e:
            logger.error(f"Error setting up explainer: {str(e)}")
            self.explainer = None
    
    def explain_prediction(self, features: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """Generate explanation for a prediction using feature importance and local analysis"""
        try:
            if self.explainer is None:
                return {"error": "Explainer not available"}
            
            # Convert input to DataFrame and prepare features
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            else:
                df = pd.DataFrame(features)
            
            # Prepare features (same as prediction pipeline)
            X = df[self.model_service.feature_names].fillna(0)
            X_scaled = self.model_service.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.model_service.model.predict(X_scaled)[0]
            probabilities = self.model_service.model.predict_proba(X_scaled)[0]
            
            # Get global feature importance from the model
            global_importance = self.model_service.model.feature_importances_
            
            # Analyze feature values relative to training data statistics
            feature_analysis = self._analyze_feature_values(X, global_importance)
            
            # Sort by combined importance score
            feature_analysis.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Get top K features
            top_features = feature_analysis[:top_k]
            
            # Generate explanations
            explanations = []
            for feat in top_features:
                if feat['deviation_score'] > 0.5:
                    if feat['feature_value'] > feat['typical_range']['high']:
                        direction = "unusually high"
                    elif feat['feature_value'] < feat['typical_range']['low']:
                        direction = "unusually low"
                    else:
                        direction = "within normal range but important"
                else:
                    direction = "typical value"
                
                explanation = f"{feat['feature']} = {feat['feature_value']:.3f} ({direction}, importance: {feat['global_importance']:.3f})"
                explanations.append(explanation)
            
            # Calculate confidence explanation
            attack_prob = probabilities[1]
            confidence_level = "HIGH" if max(probabilities) > 0.8 else "MEDIUM" if max(probabilities) > 0.6 else "LOW"
            
            return {
                'explanation_method': 'Feature Importance + Local Analysis',
                'prediction_analysis': {
                    'prediction': int(prediction),
                    'attack_probability': float(attack_prob),
                    'confidence_level': confidence_level,
                    'top_features': top_features,
                    'explanations': explanations
                },
                'interpretation': self._generate_interpretation(top_features, attack_prob, prediction),
                'feature_count': len(feature_analysis),
                'model_confidence': f"{max(probabilities):.3f}"
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': f'Explanation generation failed: {str(e)}',
                'fallback_explanation': 'Basic feature importance analysis available via /model/features endpoint'
            }
    
    def _analyze_feature_values(self, X: pd.DataFrame, global_importance: np.ndarray) -> List[Dict]:
        """Analyze feature values against typical ranges"""
        feature_analysis = []
        
        # Load training data statistics (if available)
        training_stats = self._get_training_statistics()
        
        for i, (feature_name, importance) in enumerate(zip(self.model_service.feature_names, global_importance)):
            feature_value = float(X.iloc[0, i])
            
            # Get typical range for this feature
            if training_stats and feature_name in training_stats:
                stats = training_stats[feature_name]
                typical_range = {
                    'low': stats['q25'],
                    'high': stats['q75'],
                    'mean': stats['mean'],
                    'std': stats['std']
                }
                
                # Calculate deviation score (how unusual this value is)
                if stats['std'] > 0:
                    z_score = abs((feature_value - stats['mean']) / stats['std'])
                    deviation_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1
                else:
                    deviation_score = 0.0
            else:
                # Fallback ranges
                typical_range = {'low': 0, 'high': 1, 'mean': 0.5, 'std': 0.3}
                deviation_score = 0.0
            
            # Combined score: global importance * deviation
            combined_score = float(importance) * (1 + deviation_score)
            
            feature_analysis.append({
                'feature': feature_name,
                'feature_value': feature_value,
                'global_importance': float(importance),
                'deviation_score': deviation_score,
                'combined_score': combined_score,
                'typical_range': typical_range,
                'analysis': 'unusual' if deviation_score > 0.5 else 'typical'
            })
        
        return feature_analysis
    
    def _get_training_statistics(self) -> Dict:
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
    
    def _generate_interpretation(self, top_features: List[Dict], attack_prob: float, prediction: int) -> str:
        """Generate human-readable interpretation"""
        try:
            pred_label = "ATTACK" if prediction == 1 else "NORMAL"
            confidence = "high" if attack_prob > 0.8 or attack_prob < 0.2 else "moderate"
            
            interpretation = f"This IoT network traffic is classified as {pred_label} with {confidence} confidence ({attack_prob:.1%} attack probability). "
            
            # Find most important unusual features
            unusual_features = [f for f in top_features[:3] if f['analysis'] == 'unusual']
            important_features = [f for f in top_features[:3] if f['global_importance'] > 0.1]
            
            if unusual_features:
                interpretation += f"Unusual patterns detected in: {', '.join([f['feature'] for f in unusual_features])}. "
            
            if important_features:
                interpretation += f"Key decision factors: {', '.join([f['feature'] for f in important_features])}."
            
            return interpretation
            
        except Exception as e:
            return f"IoT traffic classification: {'ATTACK' if prediction == 1 else 'NORMAL'} (confidence: {attack_prob:.1%})"
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance from the model"""
        try:
            if hasattr(self.model_service.model, 'feature_importances_'):
                importances = self.model_service.model.feature_importances_
                
                feature_summary = []
                for feature, importance in zip(self.model_service.feature_names, importances):
                    feature_summary.append({
                        'feature': feature,
                        'global_importance': float(importance)
                    })
                
                # Sort by importance
                feature_summary.sort(key=lambda x: x['global_importance'], reverse=True)
                
                return {
                    'feature_importances': feature_summary,
                    'top_5_features': feature_summary[:5],
                    'description': 'Global feature importance from Random Forest model (without SHAP)',
                    'explanation_method': 'Built-in Random Forest feature importance'
                }
            else:
                return {'error': 'Model does not support feature importance'}
                
        except Exception as e:
            logger.error(f"Error getting feature summary: {str(e)}")
            return {'error': str(e)}
    
    def get_explainer_status(self) -> Dict[str, Any]:
        """Get status of the explainer"""
        return {
            'explainer_available': self.explainer is not None,
            'explainer_type': 'Feature Importance + Local Analysis',
            'shap_available': False,  # Will be True once we fix SHAP
            'supported_explanations': ['feature_importance', 'local_analysis', 'deviation_detection'],
            'note': 'SHAP explainability will be added in next update'
        }