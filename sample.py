import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import user_template, bot_template, css
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma

# Initialize file_paths in session_state
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = []


def load_documents_from_file(file):
    # Determine the file type and use the appropriate loader
    file_extension = os.path.splitext(file)[1].lower()
    if file_extension == ".txt":
        loader = TextLoader(file)
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file)
    elif file_extension == ".docx":
        loader = UnstructuredWordDocumentLoader(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Load and return the documents
    documents = loader.load()
    return documents

def queryResult(files, query, chain_type, k):
    all_documents = []  # Create a list to store all documents

    # Load all documents from different file types
    for file in files:
        documents = load_documents_from_file(file)
        all_documents.extend(documents)

    # Split all documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(all_documents)

    # Select embeddings
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create the vector store from all documents
    db = Chroma.from_documents(texts, embeddings)

    # Expose the index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Create a chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.8), chain_type=chain_type, retriever=retriever,
        return_source_documents=True)

    # Perform the search with the query
    if query is None:
        return

    results = qa_chain({"query": query})

    # Separating the responses
    query_text = results['query']
    result_text = results['result']

    # Extracting content and metadata from source documents
    source_documents = results['source_documents']
    source_content = []
    source_metadata = []

    for doc in source_documents:
        page_content = doc.page_content
        metadata = doc.metadata
        source_content.append(page_content)
        source_metadata.append(metadata)

    # formatted_source_metadata = []
    # print(source_metadata)
    # for metadata in source_metadata:
    #     formatted_metadata = f"Page: {metadata['page']}, Source: {metadata['source']}"
    #     formatted_source_metadata.append(formatted_metadata)

    # Print or use the formatted source metadata
    # print("Formatted Source Metadata:")
    # print(formatted_source_metadata)

    return query_text, result_text, source_content[0], source_metadata[0]


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)

    st.title('Chat with Your own PDFs :books:')

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        files = st.file_uploader("Choose your Files", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner("Processing..."):
                # Save uploaded files to a temporary directory
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    file_path = file_path.replace('\\', '/')  # Replace backslashes with forward slashes
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    st.session_state.file_paths.append(file_path)  # Add the file_path to session_state

                st.write("Done")

    # Main content area
    st.header("Ask a Question")
    query = st.text_input("Enter your question here:")
    # print(st.session_state.file_paths)
    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching..."):
                response = queryResult(st.session_state.file_paths, query, "map_reduce", 1)  # Process with the user's query
            
            # Extract specific values from the response dictionary
            query_text = response[0]
            result_text = response[1]
            source_content = response[2]
            source_metadata = response[3]
            # source_metadata_2 = response[4]

            # Concatenate the extracted values
            bot_response = f"{result_text}\n\n\n{source_content}\n\n{source_metadata}\n"

            st.write(user_template.replace("{{MSG}}", query_text), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
