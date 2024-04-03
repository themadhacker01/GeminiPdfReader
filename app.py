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