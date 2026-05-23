import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", 
     "BERT: Pre-training of Deep Bidirectional Transformers",
     "GPT-3: Language Models are Few-Shot Learners",
     "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = PromptTemplate(
    template = """
Please summarize the research paper titled "{paper_input}" with the following specifications:
- Explanation Style: {style_input}
- Explanation Length: {length_input}
1. mathematical Details:
    -include relevant mathematical equations if present in the paper.
    -Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
    -Use relatable analogies to explain and simplify complex concepts and ideas in an easy-to-understand manner.
If certain information is not available in the paper, please respond with "Information not available in the paper." 
and do not attempt to generate content based on guessing.
Ensure that the summary is clear, accurate and aligned with the specified style and length requirements.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})


if st.button('Summarize'):
    result = llm.invoke(prompt)
    st.write(result.content)