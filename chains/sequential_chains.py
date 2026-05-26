#wap to get topic, send it to LLM to get detailed report, then send that report to another prompt to get a summary of the report in bullet points, give it to LLM again. Use StrOutputParser to parse the output of LLM and feed it to next prompt.

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.5,
)

prompt1 = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'From the detailed report, generate top 5 important points as summary and give in bullet points. \n {text}',
    input_variables = ['text']
)
parser = StrOutputParser()

chain = prompt1 |llm |parser | prompt2 | llm |parser

result = chain.invoke({'topic' : 'Artificial Intelligence'})
print(result)