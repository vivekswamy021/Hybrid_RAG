import os
import tempfile
import uuid  # Used to create unique IDs for each chunk inserted into Postgres

import streamlit as st  # UI Framework for rendering the web dashboard
from flashrank import Ranker, RerankRequest  # Cross-Encoder Reranker for scoring relevance
from langchain_community.document_loaders import PyMuPDFLoader  # PDF Reader/Text Extractor
from langchain_community.vectorstores import SupabaseVectorStore  # Vector DB Abstraction for LangChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # Structured Chat Roles
from langchain_groq import ChatGroq  # Groq Inference Engine wrapper for Llama 3.3
from langchain_huggingface import HuggingFaceEmbeddings  # Local Transformer Embeddings Generator
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart Text Chunking Utility
from supabase.client import Client, create_client  # Direct Supabase Client for RPC calls

# -------------------------------
# 1️⃣ Set up Environment & Page Config
# -------------------------------
st.set_page_config(page_title="Groq + Supabase True Hybrid RAG", page_icon="⚡", layout="centered")

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not groq_api_key or not supabase_url or not supabase_key:
    st.error("🚨 Missing API Keys. Please check your GROQ_API_KEY, SUPABASE_URL, and SUPABASE_KEY.")
    st.stop()

# 2️⃣ Initialize Models & DB Client
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2,
        streaming=True
    )
except Exception as e:
    st.error(f"🚨 GROQ ERROR: {e}")
    st.stop()

try:
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error(f"🚨 SUPABASE ERROR: {e}")
    st.stop()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=get_embeddings(),
    table_name="documents",
    query_name="match_documents"
)

# 3️⃣ Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

st.title("⚡ Groq & Supabase True Hybrid RAG")
st.caption("Documents uploaded here are saved permanently with Hybrid Indexing (BM25 Full-Text + Vector RRF).")

# -------------------------------
# 4️⃣ Sidebar & File Uploading
# -------------------------------
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs to the database", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Indexing {uploaded_file.name} to Supabase..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    try:
                        loader = PyMuPDFLoader(tmp_file_path)
                        docs = loader.load()
                        
                        # Optimization: Larger chunk size keeps projects/jobs/tables intact
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1200, 
                            chunk_overlap=200,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        splits = text_splitter.split_documents(docs)
                        
                        chunk_ids = [str(uuid.uuid4()) for _ in range(len(splits))]
                        vector_store.add_documents(splits, ids=chunk_ids)
                        
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path) 
    st.divider()
    if st.button("Clear Screen"):
        st.session_state.messages = []
        st.session_state.processed_files = set() 
        st.rerun()

# 5️⃣ Display Chat History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Initialize Reranker
@st.cache_resource
def get_reranker():
    return Ranker()

# -------------------------------
# 6️⃣ User Input & True Hybrid RAG Logic
# -------------------------------
user_query = st.chat_input("Type your message...")
if user_query:
    st.chat_message("user").write(user_query)
    
    # Step A: Condense chat history into a standalone search query
    search_query = user_query
    if len(st.session_state.messages) > 0:
        with st.spinner("Formulating standalone search query..."):
            condense_prompt = (
                "Given the following conversation and a follow-up question, rephrase the follow-up question "
                "to be a standalone question, in its original language, containing all necessary context.\n\n"
                f"Chat History:\n{st.session_state.messages[-4:]}\n\n"
                f"Follow-up Input: {user_query}\n\n"
                "Standalone Question:"
            )
            search_query = llm.invoke([HumanMessage(content=condense_prompt)]).content

    # Step B: True Hybrid Search (BM25 Full-Text + Vector Cosine + RRF)
    context = ""
    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(search_query)
        
        # Calls the SQL match_documents function with RRF logic
        response = supabase.rpc(
            "match_documents", 
            {
                "query_embedding": query_vector,  
                "query_text": search_query,         
                "match_count": 40  # Broad candidate pool for RRF
            } 
        ).execute()
        
        if response.data:
            st.info(f"Database retrieved {len(response.data)} candidates using RRF Hybrid Search. Reranking...")
            
            # Format database results for FlashRank
            pass_passages = [
                {
                    "id": idx,
                    "text": doc.get("content", doc.get("text", "")),
                    "meta": {"source": doc.get("metadata", {})}
                }
                for idx, doc in enumerate(response.data)
            ]
            
            # Perform FlashRank Cross-Encoder Reranking
            ranker = get_reranker()
            rerank_request = RerankRequest(query=search_query, passages=pass_passages)
            reranked_results = ranker.rerank(rerank_request)
            
            # Take top 8 post-rerank blocks
            top_n = reranked_results[:8]
            
            context = "\n\n".join([r["text"] for r in top_n])
            st.success(f"Successfully reranked down to top 8 most relevant blocks.")
            
    except Exception as e:
        st.error(f"True Hybrid Search or Reranking failed: {e}")
        
    # Step C: Construct clean LLM Input History
    messages_for_llm = []
    
    system_instruction = (
        "You are an expert document analysis assistant. Answer the user's question accurately using only "
        "the provided Context below. If the answer cannot be derived from the context, explicitly state "
        "'I cannot find that in the documents.'\n\n"
        f"Context:\n{context if context else 'No relevant context found.'}"
    )
    
    recent_messages = st.session_state.messages[-6:]
    
    messages_for_llm.append(SystemMessage(content=system_instruction))
    messages_for_llm.extend(recent_messages)
    messages_for_llm.append(HumanMessage(content=user_query))
        
    # Step D: Response Generation
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in llm.stream(messages_for_llm):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            
            st.session_state.messages.append(HumanMessage(content=user_query))
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"An error occurred: {e}")
