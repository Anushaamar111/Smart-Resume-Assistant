import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import os
os.environ["STREAMLIT_WATCHED_MODULES"] = "False"
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load .env variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Resume AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 Smart Resume AI Assistant")
st.markdown("Upload your resume and a job description to get tailored advice from the AI assistant.")

# --- Helper Function to Process Uploaded Files ---
def process_uploaded_file(uploaded_file):
    """
    Reads an uploaded file, saves it temporarily, loads its content,
    splits it into chunks, and then deletes the temporary file.
    """
    if uploaded_file is None:
        return None
    try:
        with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        return chunks
    finally:
        os.unlink(tmp_path) # Ensure the temporary file is deleted


# --- Session State Initialization ---
# This is crucial to store data across Streamlit's script reruns
if 'resume_chunks' not in st.session_state:
    st.session_state.resume_chunks = None
if 'jd_chunks' not in st.session_state:
    st.session_state.jd_chunks = None
if 'combined_db' not in st.session_state:
    st.session_state.combined_db = None

# --- File Uploaders in Columns ---
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📤 **Upload your resume (PDF or TXT):**", type=["pdf", "txt"])

with col2:
    job_description_file = st.file_uploader("📤 **Upload the job description (PDF or TXT):**", type=["pdf", "txt"])

# --- Processing Logic ---
if st.button("Analyze Resume and Job Description"):
    if resume_file and job_description_file:
        with st.spinner("Processing your resume..."):
            st.session_state.resume_chunks = process_uploaded_file(resume_file)
        st.success("✅ Resume processed successfully!")

        with st.spinner("Processing the job description..."):
            st.session_state.jd_chunks = process_uploaded_file(job_description_file)
        st.success("✅ Job description processed successfully!")

        with st.spinner("Creating a combined AI knowledge base..."):
            # Combine chunks from both documents
            combined_chunks = st.session_state.resume_chunks + st.session_state.jd_chunks

            # Create embeddings and the vector store
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.combined_db = Chroma.from_documents(combined_chunks, embedding)
        st.success("🤖 AI Assistant is ready! You can now ask questions below.")
    else:
        st.error("🚨 Please upload both your resume and the job description.")


# --- AI Chatbot Interface ---
if st.session_state.combined_db:
    st.header("💬 Talk to the AI Assistant")

    # Define a prompt template to guide the AI
    prompt_template = """
    You are an expert AI Career Coach. Your task is to analyze the user's resume in the context of the provided job description.
    Use the following pieces of context, which contain information from both the resume and the job description, to answer the user's question.
    Provide actionable advice and specific suggestions for improvement. If a question is not related to comparing the resume and job description, politely decline.

    Context:
    {context}

    Question: {question}

    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate.from_template(prompt_template)

    # Setup the RetrievalQA chain
    retriever = st.session_state.combined_db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True, # Optional: to see which chunks are being used
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    query = st.text_input("🔍 Ask a question to improve your resume (e.g., 'How can I tailor my skills section to this job?')")

    if query:
        with st.spinner("Analyzing and generating your response..."):
            try:
                response = qa_chain.invoke({"query": query})
                st.markdown("### 💡 AI-Powered Response:")
                st.write(response["result"])

                # Optional: Display the source documents the AI used for its answer
                with st.expander("See sources used by the AI"):
                    st.info("The AI used the following text chunks from your documents to generate the answer:")
                    for doc in response['source_documents']:
                        st.write(f"- {doc.page_content[:250]}...") # Display a snippet

            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.caption("⚠️ This is an AI assistant. Always review its suggestions and use your best judgment. Not a substitute for professional career advice.")

### Summary of Key Changes and Features:

# 1.  **Dual File Upload:** The UI now clearly asks for both a resume and a job description, placed in separate columns for a cleaner look.
# 2.  **State Management (`st.session_state`):** This is a critical addition. It prevents the app from losing the processed documents every time the user interacts with it, making the application efficient.
# 3.  **Combined Knowledge Base:** The core of the new functionality. The text chunks from both documents are merged *before* creating the `Chroma` vector store. This allows the retriever to search for information across both files simultaneously.
# 4.  **Action Button:** An "Analyze" button has been added to give the user explicit control over when the processing starts.
# 5.  **Enhanced Prompting (`PromptTemplate`):** This is a powerful feature that gives the AI a specific persona ("Expert AI Career Coach") and instructs it to use the context from both documents to provide actionable advice. This dramatically improves the quality and relevance of the responses.
# 6.  **Clearer User Flow:** The chatbot interface only appears after the documents have been processed, guiding the user through the steps.
# 7.  **Error Handling and Feedback:** The use of `st.spinner`, `st.success`, and `st.error` provides better feedback to the user about what the application is doing.
# 8.  **Source Documents (Optional but useful):** The code now has the capability to show you which parts of your documents the AI looked at to generate its answer, providing transparency.
