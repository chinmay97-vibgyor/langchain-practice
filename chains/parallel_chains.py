#wap to get a detailed description on a topic, from the detailed desc, create 2 things, 1. Notes, 2. Quiz questions. Do this in parallel using ParallelChain.


import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.5,
)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text. \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "generate 5 short question and answers pairs from the following text. \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "merge the provided notes and quiz into a single document. \n Notes: {notes} \n Quiz: {quiz}",
    input_variables = ["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | llm | parser,
    'quiz' : prompt2 | llm | parser
})

merge_chain = prompt3 |llm |parser

chain = parallel_chain | merge_chain

text = """ A cyber security breach occurs when an unauthorized individual gains access to a system, network, or database that holds sensitive information. These breaches can affect individuals, corporations, and even governments, causing massive financial and reputational damage. Hackers use various techniques such as phishing, malware, ransomware, and social engineering to exploit vulnerabilities in a system. Once inside, they can steal, alter, or completely destroy critical data within minutes. Personal information like passwords, credit card numbers, and social security details are among the most targeted assets. Organizations often remain unaware of a breach for weeks or even months, giving attackers enough time to cause irreversible harm. The 2017 Equifax breach, for example, exposed the personal data of nearly 147 million people worldwide. Insider threats are equally dangerous, where a disgruntled or careless employee unintentionally or deliberately leaks confidential data. Cloud misconfigurations have become a growing cause of breaches as businesses rapidly migrate their infrastructure online. Zero-day vulnerabilities — flaws unknown to the software vendor — are particularly dangerous as no patch exists at the time of exploitation. Cybercriminals often sell stolen data on the dark web, creating a thriving underground economy worth billions of dollars. Ransomware attacks have surged in recent years, where attackers encrypt an organization's data and demand payment for its release. A single breach can cost a company millions in recovery, legal fees, and regulatory fines under laws like GDPR. Multi-factor authentication, encryption, and regular security audits are essential defenses against such attacks. Employee awareness training is often considered the first line of defense, as human error remains the leading cause of breaches. Incident response planning ensures that when a breach does occur, the damage is contained quickly and efficiently. Governments worldwide are tightening cybersecurity regulations, making compliance a non-negotiable priority for businesses. Ethical hackers, also known as penetration testers, are hired to find and fix vulnerabilities before malicious actors can exploit them. As technology evolves, so do the tactics of cybercriminals, making cybersecurity a constantly shifting battlefield. Building a culture of security awareness within an organization is ultimately the strongest shield against a cyber security breach.
"""

result = chain.invoke({'text' : text})
print(result)
chain.get_graph().print_ascii()