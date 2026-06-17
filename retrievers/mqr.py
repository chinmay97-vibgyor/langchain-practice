from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

all_docs = [
    Document(page_content="Drinking enough water every day helps regulate body temperature, supports digestion, and keeps joints lubricated. Most adults are advised to drink around 2 to 3 liters daily depending on activity level.", metadata={"source": "h1"}),
    
    Document(page_content="Strength training at least twice a week helps preserve muscle mass, improve bone density, and boost metabolism. It is especially important as people age past their thirties.", metadata={"source": "h2"}),
    
    Document(page_content="A balanced diet rich in fiber, lean protein, and healthy fats supports gut health and reduces the risk of chronic diseases like diabetes and heart disease.", metadata={"source": "h3"}),
    
    Document(page_content="Getting 7 to 9 hours of quality sleep each night is essential for memory consolidation, hormone regulation, and emotional stability. Poor sleep is linked to increased stress and weight gain.", metadata={"source": "h4"}),
    
    Document(page_content="Meditation and deep breathing exercises can lower cortisol levels, reduce anxiety, and improve focus. Even 10 minutes a day of mindfulness practice can make a noticeable difference.", metadata={"source": "h5"}),
    
    Document(page_content="Cardiovascular exercises like running, cycling, and swimming strengthen the heart, improve lung capacity, and help burn calories efficiently for weight management.", metadata={"source": "h6"}),
    
    Document(page_content="Excessive sugar intake is linked to insulin resistance, weight gain, and increased risk of type 2 diabetes. Reducing processed sugar can significantly improve long term health outcomes.", metadata={"source": "h7"}),
    
    Document(page_content="Stretching before and after workouts improves flexibility, reduces muscle soreness, and helps prevent injuries during physical activity.", metadata={"source": "h8"}),
    
    Document(page_content="The Indian Premier League is one of the most watched cricket tournaments in the world, featuring franchise teams competing in a fast paced T20 format every year.", metadata={"source": "o1"}),
    
    Document(page_content="The stock market reacts to interest rate changes, inflation data, and corporate earnings reports, making it highly volatile during major economic announcements.", metadata={"source": "o2"}),
]


#initailize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# step 2: Create FAISS vector store in memory from documents
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)

similarity_retriever = vectorstore.as_retriever(
    #this enables the similarity
    search_type = "similarity",
    search_kwargs={"k":3}
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={'k':5}),
    llm = ChatGroq(model="llama-3.3-70b-versatile")
)

query = "How to improve energy balance and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content}...")

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content}...")

# output

# --- Document 1 ---
# Content: A balanced diet rich in fiber, lean protein, and healthy fats supports gut health and reduces the risk of chronic diseases like diabetes and heart disease....

# --- Document 2 ---
# Content: Drinking enough water every day helps regulate body temperature, supports digestion, and keeps joints lubricated. Most adults are advised to drink around 2 to 3 liters daily depending on activity level....

# --- Document 3 ---
# Content: Cardiovascular exercises like running, cycling, and swimming strengthen the heart, improve lung capacity, and help burn calories efficiently for weight management....
# ******************************************************************************************************************************************************

# --- Document 1 ---
# Content: Cardiovascular exercises like running, cycling, and swimming strengthen the heart, improve lung capacity, and help burn calories efficiently for weight management....

# --- Document 2 ---
# Content: Strength training at least twice a week helps preserve muscle mass, improve bone density, and boost metabolism. It is especially important as people age past their thirties....

# --- Document 3 ---
# Content: A balanced diet rich in fiber, lean protein, and healthy fats supports gut health and reduces the risk of chronic diseases like diabetes and heart disease....

# --- Document 4 ---
# Content: Stretching before and after workouts improves flexibility, reduces muscle soreness, and helps prevent injuries during physical activity....

# --- Document 5 ---
# Content: Drinking enough water every day helps regulate body temperature, supports digestion, and keeps joints lubricated. Most adults are advised to drink around 2 to 3 liters daily depending on activity level....

# --- Document 6 ---
# Content: Getting 7 to 9 hours of quality sleep each night is essential for memory consolidation, hormone regulation, and emotional stability. Poor sleep is linked to increased stress and weight gain....

# --- Document 7 ---
# Content: Meditation and deep breathing exercises can lower cortisol levels, reduce anxiety, and improve focus. Even 10 minutes a day of mindfulness practice can make a noticeable difference....

# --- Document 8 ---
# Content: Excessive sugar intake is linked to insulin resistance, weight gain, and increased risk of type 2 diabetes. Reducing processed sugar can significantly improve long term health outcomes....
