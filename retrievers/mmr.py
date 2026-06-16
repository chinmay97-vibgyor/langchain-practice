from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

documents = [
    Document(page_content = "Langchain helps you build applications with LLMs through composability. It provides a standard interface for all LLMs and a toolkit to build with them. With Langchain, you can connect your LLM to other sources of data and interact with it through a chat interface.",),
    Document(page_content = "Langchain makes it easy to build applications that leverage the power of large language models (LLMs). It provides a set of tools and abstractions that allow developers to create complex workflows and pipelines for processing natural language data. With Langchain, you can easily integrate LLMs into your applications and take advantage of their capabilities for tasks like text generation, summarization, and more.",),
    Document(page_content = "Chroma is an open-source embedding database that allows you to store and query embeddings efficiently. It provides a simple API to add, retrieve, and manage your embeddings. Chroma is designed to work seamlessly with various embedding models and can be used for tasks like semantic search, recommendation systems, and more.",),
    Document(page_content = "HuggingFace provides a wide range of pre-trained models for natural language processing tasks. Their transformers library allows you to easily use these models for tasks like text classification, named entity recognition, question answering, and more. HuggingFace also offers an extensive collection of datasets and tools to help you fine-tune and deploy your models.",),
    Document(page_content = "Vector stores are databases that are optimized for storing and querying vector representations of data. They are commonly used in applications like semantic search, recommendation systems, and machine learning. Vector stores allow you to efficiently store and retrieve high-dimensional vectors, making it easier to perform similarity searches and other operations on your data."),
    Document(page_content = "FAISS is a library for efficient similarity search and clustering of dense vectors. It provides a set of algorithms and data structures that allow you to perform fast nearest neighbor searches on large datasets. FAISS is designed to work with high-dimensional vectors and can be used for tasks like image retrieval, recommendation systems, and more.",),
    Document(page_content = "MMR (Maximal Marginal Relevance) is a technique used in information retrieval to improve the diversity of search results. It balances relevance and novelty by selecting documents that are both relevant to the query and different from each other. MMR can be used in various applications, including search engines, recommendation systems, and summarization tasks."),
    Document(page_content = "Langchain supports Chroma, FAISS, Pinecone and many more.")
]

#initailize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# step 2: Create FAISS vector store in memory from documents
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    #this enables the MMR
    search_type = "mmr",
    #k= top results, lambda_mult = relevance-diversity balance
    search_kwargs={"k":3, "lambda_mult": 1}
)

query = "What is Langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content}...")