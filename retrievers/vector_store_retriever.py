from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

#step 1: create some documents(source data)
documents = [
    Document(page_content = "Langchain helps you build applications with LLMs through composability. It provides a standard interface for all LLMs and a toolkit to build with them. With Langchain, you can connect your LLM to other sources of data and interact with it through a chat interface.",),
    Document(page_content = "Chroma is an open-source embedding database that allows you to store and query embeddings efficiently. It provides a simple API to add, retrieve, and manage your embeddings. Chroma is designed to work seamlessly with various embedding models and can be used for tasks like semantic search, recommendation systems, and more.",),
    Document(page_content = "HuggingFace provides a wide range of pre-trained models for natural language processing tasks. Their transformers library allows you to easily use these models for tasks like text classification, named entity recognition, question answering, and more. HuggingFace also offers an extensive collection of datasets and tools to help you fine-tune and deploy your models.",),
    Document(page_content = "Vector stores are databases that are optimized for storing and querying vector representations of data. They are commonly used in applications like semantic search, recommendation systems, and machine learning. Vector stores allow you to efficiently store and retrieve high-dimensional vectors, making it easier to perform similarity searches and other operations on your data."),
]

#step 2: initialise embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

#step 3: Create Chrome vector Store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

#step 4: convert vectorstore into a retriever and make a query to retrieve relevant documents based on similarity search
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content}...")