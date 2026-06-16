import wikipedia.wikipedia as wiki_internal

wiki_internal.USER_AGENT = "MyLangchainApp/1.0 (myemail@example.com)"

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")
query = "India Pakistan partition history"
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content[:1000]}...")