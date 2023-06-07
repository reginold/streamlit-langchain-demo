"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch

def load_file(file) -> list:
    """use the CSVloader to load the file"""
    loader = CSVLoader(file_path=file)
    docs = loader.load()
    return docs[1:]

def embedding(docs):
    """embed the docs"""
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
    )
    return db 

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    db = embedding(load_file("validation.csv"))
    retriever = db.as_retriever()
    llm = OpenAI(temperature=0)
    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )
    return qa_stuff



# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "chain" not in st.session_state:
    chain = load_chain()
    st.session_state["chain"] = chain

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def message(text, is_user=False, key=None):
    if is_user:
        icon = "👤"
        color = "#f0f0f0"
        align = "left"
    else:
        icon = "🤖"
        color = "#e6f7ff"
        align = "right"

    st.write(f"<div style='display: flex; align-items: center; margin-bottom: 1em; justify-content: {align};'>"
             f"<div style='font-size: 2em; margin-right: 0.5em;'>{icon}</div>"
             f"<div style='background-color: {color}; padding: 0.5em; border-radius: 1em; max-width: 70%;'>"
             f"{text}"
             f"</div>"
             f"</div>", unsafe_allow_html=True, key=key)

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = st.session_state.chain.run(query=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")