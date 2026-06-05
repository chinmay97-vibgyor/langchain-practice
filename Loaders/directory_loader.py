import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.8,
)

loader = DirectoryLoader(
    path = r'C:\langchain_genai\models_langchain\Loaders\books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load()
# docs = loader.lazy_load(), used when there a large number of documents and we want to load them one by one instead of loading all the documents at once.  

print("length : ", len(docs))
print(docs[0].page_content)
print(docs[0].metadata)