import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Document loaders and embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Model
from langchain_groq import ChatGroq

# === Chroma/protobuf compatibility check ===
try:
    import google.protobuf
    from packaging import version
    if version.parse(google.protobuf.__version__) > version.parse("3.20.1"):
        st.warning("Chroma may not work with protobuf > 3.20.x. Please run 'pip install protobuf==3.20.1 --force-reinstall' if you see errors.")
except Exception:
    pass

st.title("Conversational RAG with PDF uploads and chat history")

api_key = st.text_input(
    "Enter your Groq API key (Don't have one? Sign up at console.groq.com):", 
    type="password"
)

if api_key:
    model = ChatGroq(model="Llama3-8b-8192", groq_api_key=api_key)
    session_id = st.text_input("Enter a session ID:", value="default_session")
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        documents = []
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = f"./temp_{session_id}_{idx}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            docs = PyPDFLoader(temp_path).load()
            documents.extend(docs)
            os.remove(temp_path)

        # Split and embed
        splits = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=500
        ).split_documents(documents)
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(splits, embedding_function)
        retriever = vectorstore.as_retriever()

        # Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and latest user question which might reference context in that chat history, formulate a standalone question. Do NOT answer the question, just reformulate it if needed."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            model, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Session history management
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
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input}, 
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(f"**AI:** {response['answer']}")
            st.markdown("**Chat History:**")
            for msg in session_history.messages:
                st.markdown(f"- {msg}")
            st.markdown(f"**Session ID:** `{session_id}`")
else:
    st.warning("Please enter your Groq API key to use this app.")
