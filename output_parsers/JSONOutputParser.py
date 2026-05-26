import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.8,
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age, city, of a fictional character \n {format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction' : parser.get_format_instructions() }
)

# prompt = template.format()
# result = llm.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | llm | parser
result = chain.invoke({})
print(result)