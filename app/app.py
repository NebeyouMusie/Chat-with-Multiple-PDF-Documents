import streamlit as st
from utils import user_input, get_pdf_text, get_text_chunk, get_vector_store, get_conversational_chain

st.set_page_config(page_title="ChatPDF")

st.header("Chat with Multiple PDF Documents")
