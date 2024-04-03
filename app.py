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


def main(file_path):
    pdf_text = get_pdf_text(file_path)
    text_chunks = get_text_chunks(pdf_text)
    get_vector_store(text_chunks)

main('assets/ReoDev_GTM_Guide.pdf')