import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmailSimilarityModel:
    """Classify emails by comparing with stored emails using cosine similarity"""
    #Classification by email-to-email similarity (cosine similarity with stored emails)

    def __init__(self, emails_file="data/emails.json"):
        self.emails_file = emails_file
        try:
            with open(self.emails_file, "r") as f:
                self.stored_emails = json.load(f)  # expect list of {subject, body, embedding, ground_truth}
        except FileNotFoundError:
            self.stored_emails = []

    def _get_embedding(self, email_text: str) -> np.ndarray:
        """Convert email text into an embedding (placeholder — replace with real model)"""
        # For now: use length of text as embedding (simple numeric feature)
        return np.array([len(email_text)])

    def classify_by_similarity(self, subject: str, body: str):
        """Find the most similar stored email"""
        input_text = subject + " " + body
        input_embedding = self._get_embedding(input_text).reshape(1, -1)

        if not self.stored_emails:
            return None  # no emails stored yet

        # Compare with all stored email embeddings
        similarities = []
        for email in self.stored_emails:
            stored_embedding = np.array(email["embedding"]).reshape(1, -1)
            score = cosine_similarity(input_embedding, stored_embedding)[0][0]
            similarities.append((email, score))

        # Pick best match
        best_match, best_score = max(similarities, key=lambda x: x[1])

        return {
            "matched_email": best_match,
            "similarity_score": best_score
        }
