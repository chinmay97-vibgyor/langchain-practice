import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import TextLoader
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


loader = TextLoader(r'C:\langchain_genai\models_langchain\Loaders\doc.txt')
docs = loader.load()
# print("page content: \n", docs[0].page_content)
# print("metadata: \n", docs[0].metadata)

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.8,
)

prompt = PromptTemplate(
    template = "Write a summary of th following article in 2 lines. \n {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'text':docs[0].page_content})
print("Summary of the uploaded document: \n", result)