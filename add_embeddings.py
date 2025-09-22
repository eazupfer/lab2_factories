import json
from app.models.cosine_similarity_model import EmailSimilarityModel

# Initialize the embedding generator
similarity_model = EmailSimilarityModel()

# Load existing emails
with open("data/emails.json", "r") as f:
    emails = json.load(f)

# Add embeddings if missing
for email in emails:
    if "embedding" not in email or email["embedding"] is None:
        text = f"{email['subject']} {email['body']}"
        email["embedding"] = similarity_model._get_embedding(text).tolist()

# Save back to the file
with open("data/emails.json", "w") as f:
    json.dump(emails, f, indent=2)

print("Embeddings added for all emails.")

