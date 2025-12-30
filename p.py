import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 1. GLOBAL CONFIGURATION ---
# This ensures all functions can see the database path and API key
DB_PATH = "faiss_index_au_full"

if os.path.exists(".env"):
    load_dotenv()
    GROQ_API_KEY = os.getenv('Groq_api_key')
else:
    # Streamlit Cloud look-up
    if "Groq_api_key" in st.secrets:
        GROQ_API_KEY = st.secrets["Groq_api_key"]
    else:
        st.error("Please add 'Groq_api_key' to your Streamlit Secrets.")
        st.stop()

# --- 2. CACHED RESOURCES ---
@st.cache_resource
def load_llm_and_embeddings():
    """Initializes and caches models for maximum speed."""
    # Use the GROQ_API_KEY variable defined above
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant", 
        temperature=0.1
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# --- 3. SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the official Air University (AU) Intelligence Assistant (Pakistan).

CORE GUIDELINES:
1. IDENTITY: You are an expert on Air University Pakistan (Islamabad/Multan/Kharian). You are NOT affiliated with the US Air Force.
2. GREETINGS: If the user says 'hi', 'hello', or similar, respond warmly without using the context.
3. RAG RULES: Use the provided context to answer questions about admissions, faculty, and departments accurately. 
4. BREVITY: Be professional, concise, and use bullet points for lists.

<context>
{context}
</context>
"""

# --- 4. PROCESSING LOGIC ---
def get_response(user_input):
    llm, embeddings = load_llm_and_embeddings()
    
    # Check if index exists on the server
    if not os.path.exists(DB_PATH):
        return f"Error: The folder '{DB_PATH}' was not found on the server. Please ensure you uploaded it to GitHub.", []

    # Load the vector store
    vector_store = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 1. Immediate Greeting Check (Saves API tokens and time)
    greetings = ["hi", "hello", "hey", "salam", "aoa", "greetings"]
    if user_input.lower().strip() in greetings:
        return "Hello! I'm your Air University Assistant. How can I help you today with information about admissions, academics, or the campus?", []

    # 2. Setup Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 3. Invoke Chain with sliced history for speed
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history[-6:] 
    })
    
    return response["answer"], response["context"]

# --- 5. STREAMLIT UI ---
def main():
    st.set_page_config(page_title="AU Intelligence Portal", page_icon="🎓", layout="centered")
    
    st.markdown("<h1 style='text-align: center; color: #B38E5D;'>🎓 AU Intelligence Portal</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Initialize Session States
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display History
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Chat Input
    if user_query := st.chat_input("Ask about admissions, fees, or departments..."):
        # User side
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Assistant side
        with st.chat_message("assistant"):
            with st.spinner("Searching university database..."):
                answer, docs = get_response(user_query)
                st.markdown(answer)
                
                if docs:
                    with st.expander("View Reference Sources"):
                        for doc in docs:
                            # Safely get metadata
                            source_info = doc.metadata.get('source', 'AU Web Page')
                            st.write(f"📍 {source_info}")

        # Update History
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    main()


