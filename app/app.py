import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from utils import user_input, get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

st.set_page_config(page_title="ChatPDF", page_icon='📃')

st.header("Chat with Multiple PDF Documents")

# initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hello there👋, I can help you with your PDF's. Upload any PDF and we can chat.")
    ]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('AI'):
            st.info(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# accept user input
user_question = st.chat_input("start chatting with your pdf")
if user_question is not None and user_question != "":
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    
    with st.chat_message("Human"):
        st.markdown(user_question)
        
    response = user_input(user_question, st.session_state.chat_history)
    
    # Remove any unwanted prefixes from the response
    response = response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()
 
    with st.chat_message("AI"):
        st.write(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))


with st.sidebar:
    st.title("Menu")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

