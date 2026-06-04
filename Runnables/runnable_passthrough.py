import os
from click import prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

prompt1 = PromptTemplate(
    template = "generate a joke about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Explain the joke in simple terms. \n {text}",
    input_variables = ['text']
)

joke_generator_chain = RunnableSequence(prompt1, llm, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, llm, parser)
})

final_chain = joke_generator_chain | parallel_chain

result = final_chain.invoke({'topic': 'Dogs'})

print(result['joke'])
print(result['explanation'])