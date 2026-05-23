from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian top-order batsman known for consistency and aggressive batting.",
    "Rohit Sharma is India's opening batsman and captain, famous for elegant stroke play.",
    "Jasprit Bumrah is India's leading fast bowler known for yorkers and precise control.",
    "Shubman Gill is a young Indian opener recognized for technique and fluent batting.",
    "Hardik Pandya is an Indian all-rounder known for power hitting and seam bowling."
]

query = "tell me about jasprit"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = (cosine_similarity([query_embedding], doc_embeddings))[0]

index , score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
print (documents[index])
print (f"Similarity Score: {score}")