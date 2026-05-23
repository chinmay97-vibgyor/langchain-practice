from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "New York is a city in the United States.",
    "Paris is the capital of France.",
]

vectors = embeddings.embed_documents(documents)
for i, vector in enumerate(vectors):
    print(f"Document {i + 1}: {str(vector)}")   