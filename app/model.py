import pickle
import numpy as np


class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained model"""
        with open('models/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, features):
        """Make prediction with the model"""
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)
        probability = self.model.predict_proba(features_array)

        return {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist(),
            'class_names': ['setosa', 'versicolor', 'virginica']
        }
