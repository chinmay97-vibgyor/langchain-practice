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

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}. give results in bullet points.",
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'topic': 'Black Holes'})
print(result)

#to visualize chain:
chain.get_graph().print_ascii()