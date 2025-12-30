import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain Imports - Corrected
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage  # FIXED HERE
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 1. CONFIGURATION ---
# Replace your load_dotenv() section with this logic:
if os.path.exists(".env"):
    load_dotenv()
    GROQ_API_KEY = os.getenv('Groq_api_key')
else:
    # This will look for the secret on Streamlit Cloud
    GROQ_API_KEY = st.secrets["Groq_api_key"]

@st.cache_resource
def load_llm_and_embeddings():
    """Initializes and caches models for maximum speed."""
    llm = ChatGroq(
        groq_api_key=os.getenv('Groq_api_key'), 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# --- 2. SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the official Air University (AU) Intelligence Assistant (Pakistan).

CORE GUIDELINES:
1. IDENTITY: You are an expert on Air University Pakistan (Islamabad/Multan/Kharian). You are NOT affiliated with the US Air Force.
2. GREETINGS: If the user says 'hi', 'hello', or similar, respond warmly without using the context.
3. RAG RULES: Use the provided context to answer questions about admissions, faculty, and departments. 
4. BREVITY: Be professional, concise, and use bullet points for lists.

<context>
{context}
</context>
"""

# --- 3. PROCESSING LOGIC ---
def get_response(user_input):
    llm, embeddings = load_llm_and_embeddings()
    
    # Check if index exists
    if not os.path.exists(DB_PATH):
        return "Error: Knowledge base index not found. Please run the crawler script first.", []

    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 1. Immediate Greeting Check (Bypasses Retrieval for speed)
    greetings = ["hi", "hello", "hey", "salam", "aoa"]
    if user_input.lower().strip() in greetings:
        return "Hello! I'm your Air University Assistant. How can I help you today?", []

    # 2. Setup Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 for speed
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 3. Invoke Chain
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history[-6:] # Only send last 3 turns for speed
    })
    
    return response["answer"], response["context"]

# --- 4. STREAMLIT UI ---
def main():
    st.set_page_config(page_title="AU Intelligence", page_icon="🎓")
    st.title("🎓 AU Intelligence Portal")

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
            with st.spinner("Analyzing university data..."):
                answer, docs = get_response(user_query)
                st.markdown(answer)
                
                if docs:
                    with st.expander("View Reference Sources"):
                        for doc in docs:
                            st.write(f"📍 {doc.metadata.get('source', 'AU Web Page')}")

        # Update History
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    main()