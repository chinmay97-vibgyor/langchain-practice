import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, TextSplitter

loader = PyPDFLoader(r'C:\langchain_genai\models_langchain\Loaders\LangChain_GenAI_Curriculum.pdf')
docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=0,
    separator = ""
)

result = splitter.split_documents(docs)
print(result[0].page_content)
print(result[0].metadata)
