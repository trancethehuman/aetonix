import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 openai_api_key=os.environ["OPENAI_API_KEY"])

embeddings = OpenAIEmbeddings()

retriever = FAISS.load_local(
    folder_path="./sales-demo/aetonix/faiss_aetonix", embeddings=embeddings).as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff")

print(qa.run("how do I setup the ios app"))
