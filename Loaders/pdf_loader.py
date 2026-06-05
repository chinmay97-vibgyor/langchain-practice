import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
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

loader = PyPDFLoader(r'C:\langchain_genai\models_langchain\Loaders\LangChain_GenAI_Curriculum.pdf')
docs = loader.load()
# print(docs)
print("length : ", len(docs))
print(docs[0].page_content)
print(docs[0].metadata)