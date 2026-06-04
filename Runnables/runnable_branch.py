import os
from click import prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

def word_count(text):
    return len(text.split())

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.8,
)

prompt1 = PromptTemplate(
    template = "Generate a detailedreport on the following topic {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize the given report. \n {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

report_generator_chain = RunnableSequence(prompt1, llm, parser)
 

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 20, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough() 
)

final_chain = RunnableSequence(report_generator_chain, branch_chain)

result = final_chain.invoke({'topic': 'acid rain'})
print(result)
