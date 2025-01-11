import mailbox
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import numpy as np


def extract_emails_from_mbox(mbox_file):
    emails = []
    mbox = mailbox.mbox(mbox_file)
    for message in mbox:
        # Check if the email is multipart
        if message.is_multipart():
            for part in message.get_payload():
                if part.get_content_type() == 'text/plain':  # Extract plain-text content
                    emails.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
        else:
            # Single-part email content
            emails.append(message.get_payload(decode=True).decode('utf-8', errors='ignore'))
    return emails

def generate_embeddings(emails):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    # OR 'multi-qa-MiniLM-L6-cos-v1'
	# •	Optimized for question-answering tasks and semantic search.
	# •	Slightly better for tasks requiring detailed understanding of email text.
    embeddings = np.array(model.encode(emails)).astype(np.float32)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Embedding vector size
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings)  # Add embeddings to the index
    return index

### Main script ###

mbox_file = "mail.mbox/mbox"

# Extract emails from mbox file
print("Extracting emails from mbox file.")
emails = extract_emails_from_mbox(mbox_file)
print(f"Extracted {len(emails)} emails.")

# Generate embeddings for emails
print("Generating embeddings for emails.")
embeddings = generate_embeddings(emails)
print(f"Generated embeddings of shape {embeddings.shape}.")

# Save embeddings to file
print("Persisting embeddings to file.")
np.save("embeddings.npy", embeddings)
# Load embeddings from file
embeddings = np.load("embeddings.npy")

# Create a FAISS index
print("Creating FAISS index.")
index = create_faiss_index(embeddings)
print("FAISS index created and embeddings added.")

# Query the index
query = "Did I get any emails from cameronehrlich@gmail.com about the fires?"
query_embedding = generate_embeddings([query])[0]
distances, indices = index.search(np.array([query_embedding]), k=5)
print("Query results:")
# Print the top 5 most similar emails along with a preview of some of their content. 
for i, idx in enumerate(indices[0]):
    print(f"Email {i+1} - Similarity: {1 - distances[0][i]:.2f}")
    print(emails[idx][:500])  # Preview of the first 500 characters
    print("\n")