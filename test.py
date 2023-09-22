import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from htmlTemplates import bot_template, user_template, css
from transformers import pipeline
import docx2txt


def get_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text

def get_text_from_txt(uploaded_file):
    text = ""
    with uploaded_file as file:
        text = file.read().decode('utf-8')
    return text


def get_text_from_uploaded_files(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            text += get_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            text += get_text_from_docx(uploaded_file)
        elif file_extension == "txt":
            text += get_text_from_txt(uploaded_file)
    return text



def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    
    # For OpenAI Embeddings uncomment the below line
    # embeddings = OpenAIEmbeddings()
    
    # For Huggingface Embeddings uncomment the below line
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store):
    
    # OpenAI Model uncomment the below line

    llm = ChatOpenAI()

    # HuggingFace Model uncomment the below line
    # if you have a good machine then use 'google/flan-t5-xxl' model instead of 'WizardLM/WizardCoder-Python-34B-V1.0'
    # llm = HuggingFaceHub(repo_id="WizardLM/WizardCoder-Python-34B-V1.0", model_kwargs={"temperature":0.5, "max_length":500})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(question):

    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with Your own PDFs :books:')
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        handle_user_input(question)
    

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        files = st.file_uploader("Choose your Files and then Press OK", type=['pdf','txt','docx'], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing your PDFs..."):

                # Get text of multiple files
                raw_text = get_text_from_uploaded_files(files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
            
                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")

                # Create conversation chain
                st.session_state.conversation =  get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()