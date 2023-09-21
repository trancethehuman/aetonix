import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores import FAISS
from langsmith import Client

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "aetonix-demo"

openai_api_key = os.getenv("OPENAI_API_KEY")

# VECTOR STORE

embeddings = OpenAIEmbeddings()

retriever = FAISS.load_local(
    folder_path="./sales-demo/aetonix/faiss_aetonix", embeddings=embeddings).as_retriever()

client = Client()

st.set_page_config(
    page_title="Cats with Bats", page_icon="👩‍💻")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


with st.sidebar:
    st.write("Follow me on:")
    st.markdown(
        f'<a href="https://www.linkedin.com/in/haiphunghiem/" target="_blank"><button>LinkedIn</button></a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<a href="https://www.youtube.com/channel/UC3xGUI2ENj-86adpm-USKbg" target="_blank"><button>YouTube</button></a>',
        unsafe_allow_html=True,
    )

    st.write("Interested in alpha-testing this AI guide platform?")
    st.markdown(
        f'<a href="https://forms.gle/mnALCQQega7GzYT97" target="_blank"><button>Sign Up</button></a>',
        unsafe_allow_html=True,
    )


st.title("Aetonix Knowledge Base")
st.write("Cats with Bats")


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(
        content="I can answer any question about Aetonix based on their knowledge base. What do you want to ask?")]


# This part renders the previous messages
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# This part renders the current messages
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(temperature=0, streaming=True, callbacks=[
            stream_handler], model="gpt-3.5-turbo", openai_api_key=openai_api_key)

        qa = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, chain_type="stuff")

        response = qa(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(
            AIMessage(content=response[qa.output_key]))
        run_id = str(uuid.uuid4())

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
