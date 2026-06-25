import streamlit as st
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import tempfile
import uuid  
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# -------------------------------
# 1️⃣ Set up Environment & Page
# -------------------------------
st.set_page_config(page_title="Enterprise Gemini RAG", page_icon="🛡️", layout= "wide" if "messages" in st.session_state else "centered")

gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not gemini_api_key or not supabase_url or not supabase_key:
    st.error("🚨 Missing API Keys. Please check your environment configuration.")
    st.stop()

# -------------------------------
# 2️⃣ Initialize Enterprise Components
# -------------------------------
@st.cache_resource
def init_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=gemini_api_key,
        streaming=True,
        temperature=0.2 # Lower temperature for factual accuracy in RAG
    )

@st.cache_resource
def init_supabase():
    return create_client(supabase_url, supabase_key)

@st.cache_resource
def get_embeddings():
    # Production Upgrade: Moved from local CPU MiniLM to high-performance cloud embeddings
    return GoogleGenAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=gemini_api_key
    )

llm = init_llm()
supabase: Client = init_supabase()
embeddings = get_embeddings()

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# -------------------------------
# 3️⃣ Session State Management
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# -------------------------------
# 4️⃣ Sidebar Document Management
# -------------------------------
with st.sidebar:
    st.title("🛡️ Document Control Panel")
    st.caption("Secure enterprise document vector store.")
    
    uploaded_files = st.file_uploader(
        "Upload reference PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Processing & Indexing {uploaded_file.name}..."):
                    # Safeguard file management within block context
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        loader = PyMuPDFLoader(tmp_file_path)
                        docs = loader.load()

                        # Real-world improvement: Append structural metadata tracking
                        for doc in docs:
                            doc.metadata["source"] = uploaded_file.name
                            # In a multi-tenant environment, you would add:
                            # doc.metadata["user_id"] = current_user_id

                        # Adjusted chunk sizes for superior context preservation
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        splits = text_splitter.split_documents(docs)

                        chunk_ids = [str(uuid.uuid4()) for _ in range(len(splits))]
                        vector_store.add_documents(splits, ids=chunk_ids)

                        st.session_state.processed_files.add(uploaded_file.name)
                        st.sidebar.success(f"Indexed: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.sidebar.error(f"Failed processing {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)

    st.divider()
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_files = set() 
        st.rerun()

# -------------------------------
# 5️⃣ UI Display
# -------------------------------
st.title("Enterprise Knowledge Assistant")
st.write("Interact with your secure operational manuals, data sheets, or reports.")

# Render clean interface historical tracking
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -------------------------------
# 6️⃣ Conversational RAG Pipeline Engine
# -------------------------------
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # 1. Append and display user input immediately
    st.chat_message("user").write(user_query)
    
    # 2. Formulate Standalone Query (If chat history exists)
    search_query = user_query
    if len(st.session_state.messages) > 0:
        history_summary = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in st.session_state.messages[-4:]])
        condense_prompt = (
            f"Given the following conversation and a new user question, rewrite the question into a "
            f"standalone phrase optimized for database keyword/semantic search.\n\n"
            f"History:\n{history_summary}\n\n"
            f"New Question: {user_query}\n"
            f"Standalone Query:"
        )
        try:
            condense_response = llm.invoke([HumanMessage(content=condense_prompt)])
            search_query = condense_response.content.strip()
        except Exception:
            search_query = user_query # Fallback if rewriting fails

    # 3. Retrieve Context from Database
    context = ""
    try:
        query_vector = embeddings.embed_query(search_query)
        response = supabase.rpc(
            "match_documents", 
            {
                "query_embedding": query_vector, 
                "query_text": search_query,          
                "match_count": 5 # Consolidated count for better payload density
            } 
        ).execute()
        
        if response.data:
            # Enforce similarity threshold if your database RPC returns a similarity score
            # e.g., chunks = [d["content"] for d in response.data if d["similarity"] > 0.4]
            context = "\n\n".join([doc["content"] for doc in response.data])
            
    except Exception as e:
        st.error(f"Database extraction pipeline failure: {e}")

    # 4. Construct Executable Message Frame
    if context:
        system_instructions = (
            "You are an elite enterprise analytical AI. Answer the user's inquiry strictly utilizing the provided technical context.\n"
            "If the information is missing from the context, respond explicitly with: "
            "'I am sorry, but the requested details cannot be found in the uploaded documents.'\n\n"
            f"--- START SYSTEM CONTEXT ---\n{context}\n--- END SYSTEM CONTEXT ---"
        )
    else:
        system_instructions = "You are a helpful assistant. Advise the user that no documents matching their request have been analyzed yet."

    # Pack payload keeping real query history unpolluted by massive background injects
    execution_messages = [SystemMessage(content=system_instructions)]
    # Append past few chat messages to preserve flow without blowing up contextual memory constraints
    execution_messages.extend(st.session_state.messages[-6:])
    execution_messages.append(HumanMessage(content=user_query))

    # 5. Stream Generation and Persist
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm.stream(execution_messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Commit clean real history to session state tracking
            st.session_state.messages.append(HumanMessage(content=user_query))
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"Model Inference execution fault: {e}")
