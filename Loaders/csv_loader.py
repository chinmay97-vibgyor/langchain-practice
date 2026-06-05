from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(r'C:\langchain_genai\models_langchain\Loaders\data.csv')

data = loader.load()
print(data[5].page_content)
print(data[5].metadata)
print("length: ", len(data))
