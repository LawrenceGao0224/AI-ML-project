import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key=os.getenv("OPENAI_API_KEY")

st.set_page_config(
        page_title="PDFReader", page_icon=":bird:")    
st.header("PDFReader with QA :bird:")

file = st.file_uploader("Pick a PDF file...")
# location of the pdf file/files.
if file:    
    reader = PdfReader(file)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 3000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    message = st.text_area("What's your questions?")
    docs = docsearch.similarity_search(message)
    #chain.run(input_documents=docs, question=message)

    if message:
        st.write("Generating best practice answer...")

        result = chain.run(input_documents=docs, question=message)

        st.info(result)



