import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

st.header('Research Tool')
user_input = st.text_input("Enter your prompt")

if st.button('Summarize'):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = llm.invoke([HumanMessage(content=user_input)])
            st.write(result.content)
    else:
        st.warning("Please enter some text first.")