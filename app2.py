import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
from htmlTemplates import user_template, bot_template, css
import docx2txt
from PyPDF2 import PdfReader
from dotenv import load_dotenv

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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    # For OpenAI Embeddings uncomment the below line
    # embeddings = OpenAIEmbeddings()

    # For Huggingface Embeddings uncomment the below line
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # now we are using ChromaDB as it doesn't use system resources and faster than FAISS
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_results(question,vector_store):
    
    # OpenAI Model uncomment the below line

    llm = OpenAI()

    # HuggingFace Model uncomment the below line
    # if you have a good machine then use 'google/flan-t5-xxl' model instead of 'WizardLM/WizardCoder-Python-34B-V1.0'

    # llm = HuggingFaceHub(repo_id="WizardLM/WizardCoder-Python-34B-V1.0", model_kwargs={"temperature":0.5, "max_length":500})
    
    # Retrieve relevant documents from the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
    result = qa({"query": question})
    
    # Extract the answer
    answer = result["result"]
    
    # Extract the source documents (references)
    source_documents = result["source_documents"]
    
    # Create a list of chat messages, starting with the user's question
    chat_messages = [{"sender": "user", "content": question}]
    
    # Add the bot's answer as a chat message
    chat_messages.append({"sender": "bot", "content": answer})
    
    # Append the source documents as chat messages (customize the format as needed)
    for i, document in enumerate(source_documents):
        document_content = f"Reference {i + 1}: {document.page_content}"
        chat_messages.append({"sender": "bot", "content": document_content})
    
    # Create a result dictionary containing chat messages
    formatted_result = {"chat_messages": chat_messages}
    
    return formatted_result

# def handle_user_input(question):
#     response = {'question': question}  # Simulate user input as a dictionary
#     st.session_state.chat_history.append(response) 
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)
    
    # Check if files have been uploaded and processed
    vector_store = st.session_state.get("vector_store", None)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat_history as an empty list
    
    st.header('Chat with Your own PDFs :books:')
    
    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        files = st.file_uploader("Choose your Files and then Press OK", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing your PDFs..."):
                # Get Raw Text of multiple files
                raw_text = get_text_from_uploaded_files(files)
                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                # Create Vector Store
                vector_store = create_vector_store(text_chunks)
                st.write("DONE")
                st.session_state.vector_store = vector_store
    
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        # Ensure vector_store is available
        if vector_store is None:
            st.warning("Please upload and process your documents first.")
        else:
            # Create conversation chain
            results = get_results(question, vector_store)
            
            chat_messages = results["chat_messages"]

            for message in chat_messages:
                if message["sender"] == "user":
                    st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                elif message["sender"] == "bot":
                    st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

if __name__ == '__main__':
    main()


