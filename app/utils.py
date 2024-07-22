from config import get_google_api_key, load_config
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings 
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


load_config()
genai.configure(api_key=get_google_api_key())


# function to read pdf text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# function to chunk pdf text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# function to convert chunked text to embeddings and store locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
   
# function to setup model and chain 
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Chat history: \n{chat_history}\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
     
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model
    
    return chain


def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")    
    
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain() | StrOutputParser()
    
    response = chain.invoke({"chat_history":chat_history, "context":docs, "question":user_question})
    
    return response