import os
from click import prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Explain the joke in simple terms. \n {text}",
    input_variables = ['text']
)

chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)

result = chain.invoke({'topic': 'Cats'})
print(result)