import os
from click import prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.8,
)

prompt = PromptTemplate(
    template = "generate 5 lines on the following topic {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

notes_generator_chain = RunnableSequence(prompt, llm, parser)

parallel_chain = RunnableParallel({
    'notes' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(notes_generator_chain , parallel_chain)
result = final_chain.invoke({'topic': 'hydroelectricity'})
print("notes: ", result['notes'])
print("word count: ", result['word_count'])

