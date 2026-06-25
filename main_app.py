import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import tempfile
import uuid  
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from flashrank import Ranker, RerankRequest

# -------------------------------
# 1️⃣ Set up Environment & Page
# -------------------------------
st.set_page_config(page_title="Gemini + Supabase RAG", page_icon="🤖", layout="centered")

gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not gemini_api_key or not supabase_url or not supabase_key:
    st.error("🚨 Missing API Keys. Please check your GEMINI_API_KEY, SUPABASE_URL, and SUPABASE_KEY.")
    st.stop()
    
# 2️⃣ Initialize Models & DB Client
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=gemini_api_key,
        streaming=True
    )
except Exception as e:
    st.error(f"🚨 GEMINI ERROR: {e}")
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
    
st.title("🤖 Gemini & Supabase Hybrid RAG")
st.caption("Documents uploaded here are saved permanently to your Supabase Vector Database.")

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
                    # Fix: Write and close file to avoid WinError/Access sharing bugs
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    try:
                        loader = PyMuPDFLoader(tmp_file_path)
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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

# --- Initialize Reranker (Cache it to prevent reloading on every run) ---
@st.cache_resource
def get_reranker():
    # Downloads a small, fast model (ms-marco-MiniLM-L-6-v2) on first run
    return Ranker()

# -------------------------------
# 6️⃣ User Input & Hybrid RAG Logic
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
                f"Chat History:\n{st.session_state.messages}\n\n"
                f"Follow-up Input: {user_query}\n\n"
                "Standalone Question:"
            )
            search_query = llm.invoke([HumanMessage(content=condense_prompt)]).content

    # Step B: Vector & Keyword Retrieval (Cast a wider net for reranking)
    context = ""
    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(search_query)
        
        # 1. Fetch a larger initial candidate pool (e.g., 25 chunks)
        response = supabase.rpc(
            "match_documents", 
            {
                "query_embedding": query_vector,  
                "query_text": search_query,         
                "match_count": 25  # Wider net for the reranker to sift through
            } 
        ).execute()
        
        if response.data:
            st.info(f"Database retrieved {len(response.data)} initial candidates. Reranking...")
            
            # 2. Format database results for FlashRank
            # FlashRank expects a list of dicts with "id", "text", and optional "meta"
            pass_passages = [
                {
                    "id": idx,
                    "text": doc.get("content", doc.get("text", "")),
                    "meta": {"source": doc.get("metadata", {})}
                }
                for idx, doc in enumerate(response.data)
            ]
            
            # 3. Perform Reranking
            ranker = get_reranker()
            rerank_request = RerankRequest(query=search_query, passages=pass_passages)
            reranked_results = ranker.rerank(rerank_request)
            
            # 4. Take only the Top 5 highest-scoring chunks post-rerank
            top_n = reranked_results[:5]
            
            # Optional: Debug log to show how scores look
            # st.write([{"score": r["score"], "text": r["text"][:50]} for r in top_n])
            
            context = "\n\n".join([r["text"] for r in top_n])
            st.success(f"Successfully reranked down to the top 5 most relevant blocks.")
            
    except Exception as e:
        st.error(f"Database search or Reranking failed: {e}")
        
    # Step C: Construct clean LLM Input History
    messages_for_llm = []
    
    system_instruction = (
        "You are an expert document analysis assistant. Answer the user's question accurately using only "
        "the provided Context below. If the answer cannot be derived from the context, explicitly state "
        "'I cannot find that in the documents.'\n\n"
        f"Context:\n{context if context else 'No relevant context found.'}"
    )
    messages_for_llm.append(SystemMessage(content=system_instruction))
    messages_for_llm.extend(st.session_state.messages)
    messages_for_llm.append(HumanMessage(content=user_query))
        
    # Step D: Generate assistant response using streaming
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
