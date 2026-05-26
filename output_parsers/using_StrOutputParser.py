import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.8,
)

#1st prompt
template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ['topic']
)

#2nd prompt
template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text. /.n {text}. give results in bullet points.",
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({'topic' : 'Atomic Fission'})
print(result)

