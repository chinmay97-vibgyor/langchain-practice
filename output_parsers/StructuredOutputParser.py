import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.8,
)

schema = [
    ResponseSchema(name = 'fact_1', description = 'Fact1 about topic'),
    ResponseSchema(name = 'fact_2', description = 'Fact2 about topic'), 
    ResponseSchema(name = 'fact_3', description = 'Fact3 about topic'), 
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give 3 facts about the topic {topic} \n {format_instruction}", 
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'topic': 'Theory of Relativity'})
# result = llm.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)

chain = template | llm | parser
result = chain.invoke({'topic': 'Theory of Relativity'})
print(result)
