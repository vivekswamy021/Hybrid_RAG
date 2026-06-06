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

# -------------------------------
# 1️⃣ Set up Environment & Page
# -------------------------------
st.set_page_config(page_title="Gemini + Supabase RAG", page_icon="🤖", layout="centered")

# Get API keys from Streamlit Secrets or Environment Variables
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not gemini_api_key or not supabase_url or not supabase_key:
    st.error("🚨 Missing API Keys. Please check your GEMINI_API_KEY, SUPABASE_URL, and SUPABASE_KEY.")
    st.stop()


# -------------------------------
# 2️⃣ Initialize Models & DB Client
# -------------------------------
try:
    # 🌟 CHANGED: Using ChatGoogleGenerativeAI with Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
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
    
# ----- Embeddings -------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store connection
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=get_embeddings(),
    table_name="documents",
    query_name="match_documents"
)

# -------------------------------
# 3️⃣ Chat History & File Tracking Management
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

# TRACKING MULTIPLE FILES: Initialize a set to keep track of processed files
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
            # Check if this specific file hasn't been uploaded in the current session
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Indexing {uploaded_file.name} to Supabase..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    try:
                        loader = PyMuPDFLoader(tmp_file_path)
                        docs = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=800)
                        splits = text_splitter.split_documents(docs)

                        # Generate a list of unique strings for the ID column 
                        chunk_ids = [str(uuid.uuid4()) for _ in range(len(splits))]

                        # Push chunks into the vector store
                        vector_store.add_documents(splits, ids=chunk_ids)

                        # Mark this file as completed
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path) 

    st.divider()
    if st.button("Clear Screen"):
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        st.session_state.processed_files = set() 
        st.rerun()

# -------------------------------
# 5️⃣ Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -------------------------------
# 6️⃣ User Input & Hybrid RAG Logic
# -------------------------------

user_query = st.chat_input("Ask something about your document...")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").write(user_query)

    messages_for_llm = st.session_state.messages.copy()

    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(user_query)
        
        # Cleaned and optimize the text search for numbers/ranges
        # If search query contains numbers, isolated them so pg_tsquery doesn't choke on stop words
        search_keywords = user_query
        import re
        numbers = re.findall(r'\d+', user_query)
        if len(numbers) >= 2:
            try:
                start, end = int(numbers[0]), int(numbers[1])
                if start < end and (end - start) <= 20:
                    search_keywords = " | ".join(str(num) for num in range(start, end + 1))
            except ValueError:
                pass
                
        # Calling Hybrid Search via Supabase RPC
        response = supabase.rpc(
            "match_documents", 
            {
                "query_embedding": query_vector,  # Semantic Search Vector
                "query_text": search_keywords,         # Keyword Search Text
                "match_count": 25
            } 
        ).execute()
        
        st.info(f"Database found {len(response.data)} matching paragraphs via Hybrid Search.")
        
        # If the database returns matching context, inject it into the prompt
        if response.data:
            context = "\n\n".join([doc["content"] for doc in response.data])
            
            rag_system_prompt = (
                "You are an expert document analysis assistant. The user has uploaded files, and the text "
                "extracted from them is provided below in the Context. \n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. NEVER say you cannot read or access files. You have the file text right below.\n"
                "2. If the user asks about the documents, summarize or extract from the Context.\n"
                "3. If the user asks for a specific range of numbers, items,"
                "carefully extract exactly those items from the Context.\n"
                "4. Present them in a clean, sequentially ordered, structured list.\n"
                "5. If an item within that range is completely missing from the Context, state explicitly "
                "which ones could not be found.\n"
                "6. If the answer cannot be answered from the Context at all, say 'I cannot find that in the documents.\n"
                f"Context from uploaded files:\n{context}"
                "7. Pay close attention to numbering, indices, and specific counts within lists.'\n\n"
                f"Context from uploaded files:\n{context}"
            )
            messages_for_llm[0] = SystemMessage(content=rag_system_prompt)
            
    except Exception as e:
        # If DB fails, we log it, but messages_for_llm still exists as a normal conversation
        st.error(f"Database search failed: {e}")

    # Generate assistant response using streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # 🌟 FIX 2: Safe streaming parsing logic handled here
            for chunk in llm.stream(messages_for_llm):
                if isinstance(chunk.content, str):
                    chunk_text = chunk.content
                elif isinstance(chunk.content, list):
                    chunk_text = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in chunk.content])
                else:
                    chunk_text = str(chunk.content)

                full_response += chunk_text
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"An error occurred during LLM generation: {e}")
