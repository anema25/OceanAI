import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# --- CONFIGURATION ---
# Sets the page title and layout
st.set_page_config(page_title="AutoQA Agent", layout="wide")

# Initialize Session State to keep data across re-runs
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "html_context" not in st.session_state:
    st.session_state.html_context = ""
if "generated_test_cases" not in st.session_state:
    st.session_state.generated_test_cases = ""

# --- SIDEBAR ---
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.markdown("---")
st.sidebar.info("Upload your Product Specs and UI Guides to build the knowledge base.")

# --- BACKEND LOGIC (RAG & GEN) ---

def process_files(uploaded_files):
    """
    1. Reads uploaded files (Markdown/Text).
    2. Splits text into manageable chunks.
    3. Converts chunks into vector embeddings (numbers).
    4. Stores them in ChromaDB for retrieval.
    """
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getbuffer())
            
            # Loader selection based on file extension
            if file.name.endswith(".md"):
                loader = UnstructuredMarkdownLoader(temp_filepath)
            else:
                loader = TextLoader(temp_filepath)
            
            docs = loader.load()
            # Add metadata to track which file the info came from
            for doc in docs:
                doc.metadata["source"] = file.name
            documents.extend(docs)

    # Split text into chunks of 1000 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create Vector Store using OpenAI Embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="qa_agent_db"
    )
    return vector_store

def generate_tests_rag(query, vector_store):
    """
    1. Searches the Vector DB for text relevant to the user's query.
    2. Sends the relevant text + query to GPT-4o-mini.
    3. Returns a structured test plan.
    """
    # Retrieve top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    
    # Combine chunks into a single context string
    context_text = "\n\n".join([d.page_content for d in docs])
    sources = set([d.metadata.get("source", "Unknown") for d in docs])

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    
    prompt = f"""
    You are an expert QA Automation Engineer. 
    Based STRICTLY on the provided context, generate detailed test cases for the following request: "{query}"
    
    Context:
    {context_text}

    Output Format:
    Provide the output as a Markdown Table with these columns:
    | Test ID | Feature | Test Scenario | Expected Result | Source Doc |
    
    Do not hallucinate features not found in the text.
    """
    
    response = llm.invoke(prompt)
    return response.content, sources

def generate_selenium_script(test_case_description, html_content, vector_store):
    """
    1. Takes the specific test case and the raw HTML of the website.
    2. Asks GPT-4 to write a Python Selenium script.
    3. Ensures the script uses the actual IDs found in the HTML.
    """
    llm = ChatOpenAI(model_name="gpt-4", temperature=0) # GPT-4 is preferred for code generation
    
    prompt = f"""
    You are a Senior SDET (Software Development Engineer in Test).
    Write a complete, runnable Python Selenium script using `unittest` for the test case described below.
    
    Target HTML Structure:
    {html_content}
    
    Test Case to Automate:
    {test_case_description}
    
    Requirements:
    1. Use `webdriver.Chrome()`.
    2. Use EXPLICIT WAITS (WebDriverWait) for elements, do not use `time.sleep`.
    3. Map selectors accurately based on the provided IDs/Classes in the HTML.
    4. Include assertions to verify the Expected Result.
    5. Assume the HTML file is named 'checkout.html' and is in the same directory.
    
    Output ONLY the Python code block.
    """
    
    response = llm.invoke(prompt)
    return response.content

# --- MAIN UI ---

st.title("Autonomous QA Agent")
st.markdown("Generate Test Cases and Selenium Scripts from Documentation and HTML.")

# Create tabs for the workflow
tab1, tab2, tab3 = st.tabs(["1. Ingest Knowledge", "2. Plan Tests", "3. Generate Scripts"])

# PHASE 1: INGESTION
with tab1:
    st.header("Data Ingestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Support Documents")
        uploaded_docs = st.file_uploader("Upload .md, .txt, .json", accept_multiple_files=True)
        
    with col2:
        st.subheader("Target HTML")
        uploaded_html = st.file_uploader("Upload checkout.html", type=["html"])

    if st.button("Build Knowledge Base"):
        if not api_key:
            st.error("Please enter OpenAI API Key in Sidebar.")
        elif not uploaded_docs or not uploaded_html:
            st.error("Please upload both support docs and HTML file.")
        else:
            with st.spinner("Parsing documents and creating embeddings..."):
                # 1. Read HTML content and save to state
                html_str = uploaded_html.getvalue().decode("utf-8")
                st.session_state.html_context = html_str
                
                # 2. Build Vector DB
                try:
                    st.session_state.knowledge_base = process_files(uploaded_docs)
                    st.success("Knowledge Base Built Successfully!")
                    st.info(f"Ingested {len(uploaded_docs)} documents.")
                except Exception as e:
                    st.error(f"Error: {e}")

# PHASE 2: TEST GENERATION
with tab2:
    st.header("Test Case Generation")
    
    if st.session_state.knowledge_base is None:
        st.warning("Please build the Knowledge Base in Tab 1 first.")
    else:
        query = st.text_input("What feature do you want to test?", placeholder="e.g., Discount Codes, Form Validation")
        
        if st.button("Generate Test Cases"):
            with st.spinner("Consulting the Knowledge Base..."):
                result, sources = generate_tests_rag(query, st.session_state.knowledge_base)
                st.session_state.generated_test_cases = result
                
                st.markdown("### Generated Test Plan")
                st.markdown(result)
                st.caption(f"Sources Used: {', '.join(sources)}")

# PHASE 3: SCRIPT GENERATION
with tab3:
    st.header("Selenium Script Generator")
    
    if not st.session_state.html_context:
         st.warning("Please upload HTML in Tab 1.")
    else:
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.markdown("#### Context")
            manual_case = st.text_area("Paste a Test Case to Automate", 
                                       value="Verify that entering 'SAVE15' reduces the total price by 15%.",
                                       height=150)
        
        with col_b:
            st.markdown("#### Target")
            # Show a preview of the HTML so the user knows it's loaded
            st.code(st.session_state.html_context[:500] + "...", language="html")
            st.caption("HTML Context loaded from Tab 1")

        if st.button("Generate Python Script"):
            with st.spinner("Generating Code..."):
                script = generate_selenium_script(manual_case, st.session_state.html_context, st.session_state.knowledge_base)
                st.subheader("Generated Selenium Script")
                st.code(script, language="python")
                st.success("Copy this code to a .py file to run!")
