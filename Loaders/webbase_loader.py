import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
import os
os.environ["USER_AGENT"] = "myagent"
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.8,
)

url = 'https://www.amazon.in/Apple-2026-MacBook-Laptop-chip/dp/B0GR64G4H6/ref=sr_1_1_sspa?adgrpid=61545516840&dib=eyJ2IjoiMSJ9.sFrIQV35WPTsuQ8KCaPXibjVVjgLqj3YnD7YQWxOVacVgGsjqn11pkekS8vVXTax1ao3XOibbmKhr8stZX55Q2ukto_WFRt8qon-YFDONtxiAvB3SlZghoduTfU7RxNoxkDS3xBvEDLvFiUYqdEFB4FJzB6G-w47vWwj34cGKfqvq2QbPlQ8HVtuf2BtNxaqdmvt_lZ9lzIhnbbfu08YTB69p2tRY6qPedRknzteUv4.ZQrK_tyalcSOLyXrmh06Lq1573CWJRZ-2IlRL5yt_Sw&dib_tag=se&gad_source=1&hvadid=398125214057&hvdev=c&hvexpln=0&hvlocphy=9062000&hvnetw=g&hvocijid=9473184189984609829--&hvqmt=e&hvrand=9473184189984609829&hvtargid=kwd-2366115905272&hydadcr=26915_2178229&keywords=macbook%2Bair%2Bm2%2Bprice%2Bamazon&mcid=e360639ba1bc3dce8ffaf66f5f6ae3bb&qid=1780657412&sr=8-1-spons&aref=WQDYl9QlIb&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'
loader = WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(
    template = "Answer the following question \n {question} from the \n {text}",
    input_variables = ['question', 'text' ]
)
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'question': 'What is the product that is being sold?', 'text': docs[0].page_content})

print(result)
