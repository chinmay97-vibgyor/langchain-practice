from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content=(
        """
        Solar panels convert sunlight into electricity using photovoltaic cells.
        The cost of installation has dropped significantly over the past decade, making
        residential solar more affordable. Most homeowners see a return on investment
        within 6 to 8 years through reduced electricity bills. Government subsidies in
        many countries further lower the upfront cost. Maintenance is minimal since panels
        have no moving parts and typically last 20 to 25 years. Cloudy weather reduces
        efficiency but does not stop power generation entirely. Battery storage systems
        are often paired with solar panels to provide power during the night.
        """
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """
        The Great Wall of China was built over several centuries by different dynasties
        to protect against invasions from northern tribes. It stretches over 13,000 miles
        and is considered one of the most impressive feats of ancient engineering. Construction
        began as early as the 7th century BC, with major expansions during the Ming Dynasty.
        Millions of workers, including soldiers, peasants, and prisoners, contributed to its
        construction. Today it stands as a UNESCO World Heritage Site and a major tourist
        attraction, drawing millions of visitors every year.
        """
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """
        Regular exercise improves cardiovascular health by strengthening the heart muscle
        and improving blood circulation. It also helps regulate blood sugar levels, reducing
        the risk of type 2 diabetes. Beyond physical benefits, exercise releases endorphins
        that improve mood and reduce symptoms of anxiety and depression. Experts recommend
        at least 150 minutes of moderate aerobic activity per week. Strength training twice
        a week is also advised to maintain muscle mass, especially as people age. Even short
        walks can provide noticeable health benefits when done consistently.
        """
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """
        Artificial intelligence is being increasingly used in healthcare to assist with
        diagnostics, drug discovery, and personalized treatment plans. Machine learning
        models can analyze medical images such as X-rays and MRIs to detect abnormalities
        faster than traditional methods in some cases. AI-powered chatbots are also helping
        hospitals manage patient inquiries and appointment scheduling. However, concerns
        remain around data privacy, algorithmic bias, and the need for human oversight in
        critical medical decisions. Regulatory bodies are working to establish guidelines
        for safe and ethical AI deployment in healthcare settings.
        """
    ), metadata={"source": "Doc4"}),

    Document(page_content=(
        """
        The history of jazz music traces back to the late 19th and early 20th centuries
        in New Orleans, blending African rhythms, blues, and European musical structures.
        Improvisation is a defining feature of jazz, allowing musicians to create unique
        performances each time a piece is played. Legendary figures like Louis Armstrong
        and Duke Ellington helped popularize the genre across the United States and eventually
        the world. Jazz later influenced numerous other genres including rock, hip hop, and
        funk. Festivals dedicated to jazz continue to celebrate its rich cultural heritage today.
        """
    ), metadata={"source": "Doc5"}),
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# step 2: Create FAISS vector store in memory from documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#setup compressor using an LLM

llm = ChatGroq(model="llama-3.3-70b-versatile")
compressor = LLMChainExtractor.from_llm(llm)

#create contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "How does AI help in medical diagnosis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content}...")


# output

# --- Document 1 ---
# Content: Artificial intelligence is being increasingly used in healthcare to assist with 
#         diagnostics, drug discovery, and personalized treatment plans. Machine learning 
#         models can analyze medical images such as X-rays and MRIs to detect abnormalities 
#         faster than traditional methods in some cases.