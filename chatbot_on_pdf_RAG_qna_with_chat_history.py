# config
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# prompts
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

# models
import langchain_google_genai
import langchain_ollama
import langchain_openai
from langchain_groq import ChatGroq
model = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)

# output parsers
from langchain_core.output_parsers import StrOutputParser

# message history
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory, SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory  
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

#> RAG
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#  Streamlit
import streamlit as st


# st.title("Conversational RAG with PDF uploads and chat history")
# st.write("Upload your PDFs and chat with them!")
# api_key = st.text_input("Enter your Groq API key (Don't have one, sign up here: https://console.groq.com):", type="password")
# if api_key:
#     model = ChatGroq(model="Llama3-8b-8192", groq_api_key=api_key)
#     session_id = st.text_input("Enter a session ID:", value="default_session")
#     if "store" not in st.session_state:
#         st.session_state.store = {}
    
#     uploaded_files = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True)
#     if(uploaded_files):
#         documents = []
#         for uploaded_file in uploaded_files:
#             temppdf= f"./temp.pdf"
#             with open(temppdf, "wb") as file:
#                 file.write(uploaded_file.read())
#                 file_name = uploaded_file.name

#             docs = PyPDFLoader(temppdf).load()
#             documents.extend(docs)
#             # os.remove(temppdf)
#         splits = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500).split_documents(documents)
#         vectorstore = Chroma.from_documents(splits, embedding_function)
#         retriever = vectorstore.as_retriever()


#         contextualize_q_prompt = ChatPromptTemplate.from_messages([
#             ("system",(
#             "Given a chat history and lastest user question"
#             "which might refernce context in that chat history,"
#             "formulate a standalone question which can be understood"
#             "without the chat history. Do NOT answer the question,"
#             "just reformulate it if needed and otherwise return it as is."
#             )),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human","{input}"),
#         ])
#         history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

#         qa_prompt = ChatPromptTemplate.from_messages([
#             ("system", (
#             "You are an assistant for question-answering tasks."
#             "Use the following pieces of retrieved context to answer"
#             "the question. If you don't know the answer, say that you"
#             "don't know, don't try to make up an answer."
#             "Use three sentences maximum and keep the answer concise."
#             "\\n\\n"
#             "{context}"
#         )),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}")
#         ])

#         qa_chain = create_stuff_documents_chain(model, qa_prompt)
#         rag_chain = create_retrieval_chain( history_aware_retriever, qa_chain)


#         def  get_session_history(session_id):
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id] = ChatMessageHistory()
            
#             return st.session_state.store[session_id]
        
#         conversational_rag_chain = RunnableWithMessageHistory(qa_chain, get_session_history=get_session_history, input_messages_key="input", output_messages_key="answer", history_messages_key="chat_history")
#         chat = model.chat(history=history, prompt=chat_prompt, temperature=0.7, max_tokens=1024)
#         user_input = st.text_input("You:", key="input")
#         if(user_input):
#             session_history=get_session_history(session_id)
#             response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable":{"session_id": session_id}})
#             st.write("AI:",response["answer"])

#             st.success("Chat History:", session_history.messages)
#             st.write("Session ID:", session_id)
#             st.write("Session History:", st.session_state.store)

# else:
#     st.warning("Please enter your Groq API key to use this app.")

st.title("Conversational RAG with PDF uploads and chat history")

api_key = st.text_input("Enter your Groq API key (Don't have one, sign up here: https://console.groq.com):", type="password")
if api_key:
    model = ChatGroq(model="Llama3-8b-8192", groq_api_key=api_key)
    session_id = st.text_input("Enter a session ID:", value="default_session")
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = f"./temp_{idx}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            docs = PyPDFLoader(temp_path).load()
            documents.extend(docs)
            os.remove(temp_path)

        splits = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500).split_documents(documents)
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(splits, embedding_function)
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and latest user question which might reference context in that chat history, formulate a standalone question. Do NOT answer the question, just reformulate it if needed."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_session_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )

        user_input = st.text_input("You:", key="input")
        if (user_input):
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable":{"session_id": session_id}})
            st.write    ("AI:",response["answer"])

            st.write("Chat History:", session_history.messages)
            st.write("Session ID:", session_id)
            st.write("Session History:", st.session_state.store)
else:
    st.warning("Please enter your Groq API key to use this app.")