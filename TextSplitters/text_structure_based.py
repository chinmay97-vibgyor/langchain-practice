from langchain_text_splitters import CharacterTextSplitter, TextSplitter, RecursiveCharacterTextSplitter

text = """
Text splitting is a crucial step in natural language processing (NLP) tasks, especially when dealing with large documents or corporations. It involves breaking down a text into smaller, more manageable pieces, such as sentences, paragraphs, or even individual words. This process is essential for various applications, including text analysis, information retrieval, and machine learning. There are several methods for text splitting, each with its own advantages and use cases. One common approach is to split text based on punctuation marks, such as periods, commas, or semicolons. This method is effective for separating sentences and can be useful for tasks like sentiment analysis or topic modeling. Another approach is to split text based on whitespace, which can be useful for tokenization and word-level analysis. Additionally, more advanced techniques, such as using regular expressions or machine learning models, can be employed to split text based on specific patterns or linguistic features. Overall, text splitting is a fundamental step in NLP that enables more efficient processing and analysis of textual data, ultimately leading to better insights and improved performance in various applications.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(chunks)
