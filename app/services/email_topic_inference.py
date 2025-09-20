from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

#added code
import json
from pathlib import Path
TOPICS_FILE = Path("data/topic_keywords.json")
#end added code


class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
        #NEW CODE Load topics from JSON file
        try:
            with open(TOPICS_FILE, "r") as f:
                self.model.topics = json.load(f)
        except FileNotFoundError:
            self.model.topics = []  # start empty if file doesn't exist
            ###END NEW CODE
    
    def classify_email(self, email: Email) -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        # Step 2: Classify using features
        predicted_topic = self.model.predict(features)
        topic_scores = self.model.get_topic_scores(features)
        
        # Return comprehensive results
        #ADDED NEW RETURN CODE
        return {"available_topics":self.model.topics}
            # "predicted_topic": predicted_topic,
            # "topic_scores": topic_scores,
            # "features": features,
            # "available_topics": self.model.topics,
            # "email": email
        #}
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }