import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import os
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Load .env variables
load_dotenv()


st.set_page_config(page_title="Smart Resume Assistant")
st.title("📄 Smart Resume Assistant")

# Upload file
uploaded_file = st.file_uploader("📤 Upload your resume (PDF or TXT):", type=["pdf", "txt"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()

    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Vector embedding with explicit model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma vector store (auto-persistence in Chroma v0.4+)
    vectordb = Chroma.from_documents(chunks, embedding, persist_directory="db")

    # Retrieval QA setup
    retriever = vectordb.as_retriever()
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Resume processed! Ask your questions now.")

    query = st.text_input("🔍 Ask something about your resume (e.g. 'Improve my summary'):")

    if query:
        with st.spinner("Analyzing your resume..."):
            response = qa_chain.invoke({"query": query})
        st.markdown("### 💡 Response:")
        st.write(response["result"])

    st.caption("⚠️ This is an AI assistant, not a substitute for professional career advice.")
