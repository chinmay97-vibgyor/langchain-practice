#WAP to take a feedback from user, send it to LLM to get the sentiment of the feedback, if the sentiment is negative, give it back to the LLM to generate an appropriate reply for negative feedback and the same for positive feedback. Use StrOutputParser to parse the output of LLM and feed it to next prompt.

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following text into positive or negative. \n {feedback} \n {format_instruction}",
    input_variables = ["feedback", "format_instruction"],
    partial_variables = {"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | llm | parser2

prompt2 = PromptTemplate(
    template = "Write an appropriate response for this positive feedback: {feedback}",
    input_variables = ["feedback"]
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response for this negative feedback: {feedback}",
    input_variables = ["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | llm | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | llm | parser),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback': "The service was excellent! I am very happy with the support I received."})
print(result)
chain.get_graph().print_ascii()
