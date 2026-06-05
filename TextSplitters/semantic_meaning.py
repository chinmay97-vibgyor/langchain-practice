from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0,
)

sample = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. 
The term was coined by John McCarthy in 1956 during the Dartmouth Conference, which is considered the birth of AI as a field. 
Early AI research focused on problem solving and symbolic methods. 
Scientists believed that human intelligence could be precisely described and simulated by machines.
In the 1970s and 1980s, AI research faced significant funding cuts known as the AI winters. 
Progress was slower than expected and many promises made by researchers could not be delivered. 
However, interest in the field was revived in the 1990s with the rise of machine learning approaches. 
Researchers shifted focus from rule-based systems to systems that could learn from data.

Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. 
It focuses on developing algorithms that can access data and use it to learn for themselves. 
Supervised learning, unsupervised learning, and reinforcement learning are the three main types of machine learning. 
In supervised learning, models are trained on labeled data to make predictions on new unseen data.
Decision trees, support vector machines, and linear regression are classic machine learning algorithms. 
These algorithms work well on structured tabular data but struggle with unstructured data like images and text. 
Feature engineering was a critical and time consuming step in traditional machine learning pipelines. 
Data scientists had to manually design features that would help the model learn effectively.

Deep learning is a subset of machine learning that uses neural networks with many layers to learn from large amounts of data. 
Inspired by the structure of the human brain, deep neural networks can automatically learn features from raw data. 
Convolutional Neural Networks or CNNs revolutionized the field of computer vision and image recognition. 
Recurrent Neural Networks or RNNs were designed to handle sequential data like text and time series.
The breakthrough moment for deep learning came in 2012 when AlexNet won the ImageNet competition by a large margin. 
This demonstrated that deep neural networks could significantly outperform traditional computer vision methods. 
GPUs played a crucial role in making deep learning practical by enabling fast parallel computation. 
Frameworks like TensorFlow and PyTorch made it easier for researchers to build and train deep learning models.

Natural Language Processing or NLP is the branch of AI that deals with the interaction between computers and human language. 
Early NLP systems relied on hand-crafted rules and linguistic knowledge to process text. 
Statistical methods later replaced rule-based approaches and improved performance on many language tasks. 
Tasks like machine translation, sentiment analysis, and named entity recognition became standard NLP benchmarks.
The introduction of word embeddings like Word2Vec in 2013 was a major milestone in NLP. 
Words were represented as dense vectors that captured semantic relationships between them. 
This allowed models to understand that words like king and queen are related in meaning. 
Transfer learning further transformed NLP by allowing models pretrained on large corpora to be fine-tuned for specific tasks.

Transformers are a type of deep learning architecture introduced in the paper Attention Is All You Need in 2017. 
They rely on a mechanism called self-attention to process input sequences in parallel rather than sequentially. 
This made training significantly faster and allowed models to capture long range dependencies in text. 
BERT and GPT are two of the most influential transformer based models in the history of NLP.
GPT or Generative Pretrained Transformer by OpenAI is designed to generate coherent and contextually relevant text. 
It is trained on massive amounts of internet text and learns to predict the next word in a sequence. 
GPT-3 released in 2020 had 175 billion parameters and demonstrated remarkable few-shot learning capabilities. 
ChatGPT built on top of GPT models became a global phenomenon after its release in November 2022.

AI has found applications across a wide range of industries transforming the way businesses operate. 
In healthcare, AI is used for medical image analysis, drug discovery, and personalized treatment recommendations. 
Models trained on thousands of medical scans can detect diseases like cancer with accuracy comparable to human doctors. 
AI powered chatbots are being used in hospitals to assist patients and reduce the burden on medical staff.
In the financial sector, AI is used for fraud detection, algorithmic trading, and credit risk assessment. 
Banks use machine learning models to analyze transaction patterns and flag suspicious activities in real time. 
Robo-advisors powered by AI provide personalized investment advice to retail investors at low cost. 
AI is also transforming the insurance industry through automated claims processing and risk prediction.

Despite its many benefits, AI also raises serious ethical concerns that need to be addressed. 
Bias in AI systems is one of the most discussed issues, where models trained on biased data produce unfair outcomes. 
Facial recognition systems have been shown to perform poorly on darker skin tones due to lack of diverse training data. 
Algorithmic hiring tools have been found to discriminate against women and minority candidates in some cases.
Privacy is another major concern as AI systems often require large amounts of personal data to function effectively. 
Surveillance systems powered by AI can track individuals without their knowledge or consent. 
The use of AI in autonomous weapons raises questions about accountability and the rules of war. 
Governments and organizations around the world are working on AI regulations and ethical frameworks to address these issues.

The future of AI holds enormous promise but also significant challenges. 
Artificial General Intelligence or AGI refers to AI systems that can perform any intellectual task that a human can. 
Most researchers believe that AGI is still decades away and requires fundamental breakthroughs in our understanding of intelligence. 
Some experts like Elon Musk and Nick Bostrom have warned about the existential risks posed by superintelligent AI.
On the positive side, AI has the potential to solve some of humanity's greatest challenges. 
It could accelerate scientific discovery, help combat climate change, and eradicate diseases. 
AI powered education platforms can provide personalized learning experiences to students around the world. 
The key to realizing this potential lies in developing AI that is safe, transparent, and aligned with human values.
"""

docs = text_splitter.create_documents([sample])
print("length: ", len(docs))
print(docs)