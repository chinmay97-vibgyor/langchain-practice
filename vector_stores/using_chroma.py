from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document

doc1 = Document(
    page_content="Virat Kohli is one of the greatest batsmen in IPL history. He has scored over 8000 runs for Royal Challengers Bengaluru. He is the only player to score 100 centuries in international cricket. His consistency and aggressive batting make him a nightmare for bowlers.",
    metadata={"team": "Royal Challengers Bengaluru", "role": "Batsman"}
)

doc2 = Document(
    page_content="MS Dhoni is the most legendary captain in IPL history. He has led Chennai Super Kings to five IPL titles. Known for his ice cool finishing and lightning fast stumping, he is the heartbeat of CSK. Fans across India call him Thala with deep respect and love.",
    metadata={"team": "Chennai Super Kings", "role": "Wicket Keeper"}
)

doc3 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history with five titles for Mumbai Indians. He is famous for his elegant pull shots and effortless stroke play. Rohit is known for scoring big centuries when his team needs it most. He is widely regarded as one of the best white ball batsmen in the world.",
    metadata={"team": "Mumbai Indians", "role": "Batsman"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is the most feared fast bowler in the IPL. He is known for his unplayable yorkers and unpredictable slower balls at the death. Bumrah has been the backbone of Mumbai Indians bowling attack for years. His unique slingy action makes him extremely difficult to pick for any batsman.",
    metadata={"team": "Mumbai Indians", "role": "Bowler"}
)

doc5 = Document(
    page_content="Rishabh Pant is one of the most explosive wicket keeper batsmen in the IPL. He plays for Delhi Capitals and is known for his unorthodox yet effective batting style. Pant can turn a match on its head within just a few overs. His fearless approach and big hitting ability make him a match winner on any given day.",
    metadata={"team": "Delhi Capitals", "role": "Wicket Keeper"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    persist_directory="my_chroma_db",
    collection_name="ipl_players"
)

#add_docsuments to the vector store
ids = vector_store.add_documents(docs)
print(ids)

result = vector_store.get(include=["embeddings", "metadatas", "documents"], ids=ids)

for i, id in enumerate(result["ids"]):
    print(f"\n--- Doc {i+1} ---")
    print(f"ID       : {id}")
    print(f"Document : {result['documents'][i]}")
    print(f"Metadata : {result['metadatas'][i]}")
    print(f"Embedding: {result['embeddings'][i][:5]}...")  # first 5 dims only

results = vector_store.similarity_search_with_score(
    query="Who among these is a wicket keeper?",
    k=2,
    filter={"role": "Wicket Keeper"}
)

for doc, score in results:
    print(f"Document: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print("---")