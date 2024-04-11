import os
import streamlit as st
import google.generativeai as genai

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Loads .env file
load_dotenv()

# Loads the gen ai API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key = GOOGLE_API_KEY)


# Extract text from pdf with file path as input
def get_pdf_text(file_path):
    pdf_text = ''

    # Open the pdf file with read permissions in binary format
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)

        # Iterate through each page in the pdf reader pages object
        for pdf_page in pdf_reader.pages:
            pdf_text += pdf_page.extract_text()

        # Return the concatenated text from all pdf pages
        return pdf_text


# Splits the large document into smaller chunks using LangChain
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    text_chunks = text_splitter.split_text(pdf_text)

    # Returns the text chunks from the pdf
    return text_chunks


# Creating a vector store using FAISS
def get_vector_store(text_chunks):
    # Create embeddings using Google GenAI model
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

    # Create a vector store using FAISS from the text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)

    # Save the vector store locally with the name faiss_index
    vector_store.save_local('faiss_index')


# Develop a question-answer chain
def get_conversational_chain():
    # Define a prompt template for asking qns based on a given context
    prompt_template = '''
    Answer the question as detailed as possible from the provided context, make sure to provide all the details
    If the answer is not in the provided context, "Answer is not available in the context"
    No matter what, do not provide the wrong answer
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    '''

    # Initialise a ChatGoogleGenerateAI model for conversational AI
    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.3)

    # Create a prompt template with input variables context, question
    prompt = PromptTemplate(
        template = prompt_template,
        input_variables = ['context', 'question']
    )

    # Load a question-answering chain with the model, prompt
    chain = load_qa_chain(model, chain_type = 'stuff', prompt = prompt)

    return chain


# Take user input
def user_input(user_question):
    # Create embedding for the user question using a Google GenAI model
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

    # Load a FAISS vector database from a local file
    # The key allow_dangerous_deserialization = True is a security risk
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization = True)

    # Perform similarity search in the vector database based on the user qn
    docs = new_db.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the chain to get a response for the user question and documents
    response = chain(
        {
            'input_documents': docs,
            'question': user_question
        },
        # return_only_outputs = True
    )

    # Display the response in a Streamlit app
    st.write('Reply : ', response['output_text'])


# Main function of the application that calls all other functions
def main(file_path):
    st.set_page_config('Chat PDF')
    st.header('Chat with PDF using Gemini')

    user_question = st.text_input('Ask a question from the PDF files')

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title('Status : ')
        pdf_text = get_pdf_text(file_path)
        text_chunks = get_text_chunks(pdf_text)
        get_vector_store(text_chunks)
        st.success('Done')


if __name__ == '__main__':
    main('assets/sample_file.pdf')