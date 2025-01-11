import mailbox
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sys
import logging
import re

def extract_emails_from_mbox(mbox_file):
    emails = []
    mbox = mailbox.mbox(mbox_file)
    for message in mbox:
        email_data = {
            "subject": message["subject"],
            "sender": message["from"],
            "timestamp": message["date"],
            "body": ""
        }
        # Check if the email is multipart
        if message.is_multipart():
            for part in message.get_payload():
                if part.get_content_type() == 'text/plain':  # Extract plain-text content
                    email_data["body"] += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            email_data["body"] = message.get_payload(decode=True).decode('utf-8', errors='ignore')
        emails.append(email_data)
    return emails

def preprocess_email_content(content):
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Normalize whitespace
    content = ' '.join(content.split())
    return content

def split_long_email(content, max_length=512):
    words = content.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def generate_email_embeddings(emails, model_name='multi-qa-MiniLM-L6-cos-v1', max_length=512, batch_size=32):
    try:
        # Load the model
        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        # Preprocess and chunk email content
        email_chunks = []
        chunk_to_email_mapping = []  # Maps chunks to original email indices
        for idx, email in enumerate(emails):
            preprocessed_email = preprocess_email_content(email)
            chunks = split_long_email(preprocessed_email, max_length=max_length)
            email_chunks.extend(chunks)
            chunk_to_email_mapping.extend([idx] * len(chunks))  # Map each chunk to the email index

        print(f"Generating embeddings for {len(email_chunks)} chunks...")

        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(email_chunks), batch_size):
            batch = email_chunks[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{-(-len(email_chunks) // batch_size)}")
            batch_embeddings = model.encode(batch, convert_to_tensor=False)
            embeddings.append(batch_embeddings)

        # Combine and convert to float32
        embeddings = np.vstack(embeddings).astype(np.float32)
        print("Embeddings generated successfully.")
        return embeddings, chunk_to_email_mapping

    except Exception as e:
        print(f"Error generating email embeddings: {e}")
        raise RuntimeError("Failed to generate email embeddings.") from e

def generate_query_embedding(query, modle_name='multi-qa-MiniLM-L6-cos-v1'):
    model = SentenceTransformer(modle_name)
    return model.encode(query).astype(np.float32)

def create_faiss_index(embeddings):
    # Normalize embeddings to unit vectors for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def main():
    # Specify the mbox file
    # model = "multi-qa-MiniLM-L6-cos-v1"
    model = 'all-MiniLM-L6-v2'
    mbox_file = "mail.mbox/mbox"

    # Extract emails from mbox file
    print("Extracting emails from mbox file.")
    emails = extract_emails_from_mbox(mbox_file)
    print(f"Extracted {len(emails)} emails.")

    # Generate embeddings for emails
    print("Generating embeddings for emails.")
    embeddings, chunk_to_email_mapping = generate_email_embeddings([email['body'] for email in emails], model)
    print(f"Generated embeddings of shape {embeddings.shape}.")

    # Save embeddings to file
    print("Persisting embeddings to file.")
    np.save("embeddings.npy", embeddings)

    # Create a FAISS index
    print("Creating FAISS index.")
    index = create_faiss_index(embeddings)
    print("FAISS index created and embeddings added.")

    # Real-time query loop
    print("Enter your query below (type 'exit' to quit):")
    while True:
        query = input("Query: ").strip()
        if query.lower() == 'exit':
            print("Exiting query tool.")
            break
        
        # Generate the embedding for the query
        query_embedding = generate_query_embedding(query, model)
        distances, indices = index.search(np.array([query_embedding]), k=5)

        # Display the top 5 most similar emails with their similarity scores
        print("Found:")
        for i, idx in enumerate(indices[0]):
            email_idx = chunk_to_email_mapping[idx]  # Map chunk index back to original email
            print(f"Email {i+1} - Similarity: {distances[0][i]:.2f}")
            print(f"Subject: {emails[email_idx]['subject']}")
            print(f"Sender: {emails[email_idx]['sender']}")
            print("\n")

        # Display only results above the similarity threshold
        print("Query results:")
        # Trim indices so that we only get the top re
        similarity_threshold = 0.40 # TODO: Play with the similarity threshold
        has_results = False
        for i, idx in enumerate(indices[0]):
            similarity = distances[0][i]
            if similarity >= similarity_threshold:
                has_results = True
                email_idx = chunk_to_email_mapping[idx]
                print(f"Email {i+1} - Similarity: {similarity:.2f}")
                print(f"Subject: {emails[email_idx]['subject']}")
                print(f"Sender: {emails[email_idx]['sender']}")
                print(f"Body Preview: {emails[email_idx]['body'][:300]}")
                print("\n")
        
        if not has_results:
            print(f"No results found with similarity above {similarity_threshold:.2f}.\n")

if __name__ == "__main__":
    main()