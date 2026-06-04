import os
from click import prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

prompt1 = PromptTemplate(
    template = "generate a tweet about {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "generate a LinkedIn post about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, llm, parser),
    'linkedin_post': RunnableSequence(prompt2, llm, parser)
})

result = parallel_chain.invoke({"topic" : "Unemployment in India"})
print(result['tweet'])
print(result['linkedin_post'])