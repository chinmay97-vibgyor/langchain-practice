import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description= "name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables = ['place'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)


# prompt = template.invoke({'place': 'Russian'})
# print('prompt:', prompt)

# result = llm.invoke(prompt)
# final_result = parser.parse(result.content)

chain  = template | llm | parser
final_result = chain.invoke({'place': 'Indian'})
print(final_result)