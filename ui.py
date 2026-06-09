import streamlit as strl
import requests
from pypdf import PdfReader

BACKEND_API = "http://127.0.0.1:8000"

strl.set_page_config(page_title="Aarav Knowledge Interface", layout="wide")

# Sidebar Tracking Module Control
strl.sidebar.title("Navigation Hub")
app_mode = strl.sidebar.radio("Select Processing Interface", ["Document Ingestion Workspace", "Conversational Chat Pipeline"])

# --- MODE 1: DOCUMENT INGESTION CONSOLE ---
if app_mode == "Document Ingestion Workspace":
    strl.title("🗂 Document Ingestion & Spatial Embedding Console")
    strl.write("Parse arbitrary native digital PDF files and compile their contents into the local `pgvector` store.")
    
    uploaded_file = strl.file_uploader("Upload Target PDF Document", type=["pdf"])
    
    if uploaded_file is not None:
        strl.success(f"File handle loaded successfully: {uploaded_file.name}")
        
        if strl.button("Compile & Generate Embeddings"):
            with strl.spinner("[*] Extracting digital character tracking streams..."):
                try:
                    # Extract text from uploaded PDF memory stream
                    reader = PdfReader(uploaded_file)
                    extracted_text = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text.append(text)
                    
                    full_raw_text = "\n".join(extracted_text)
                    
                    if not full_raw_text.strip():
                        strl.error("Extraction error: File contains zero token sequences or is an un-OCRed image scan.")
                    else:
                        # Push payload to backend API
                        payload = {"filename": uploaded_file.name, "text_content": full_raw_text}
                        response = requests.post(f"{BACKEND_API}/ingest", json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            strl.success(f"[+] Ingestion vector compilation successful. Wrote {data['chunks_ingested']} context blocks to pgvector.")
                        else:
                            strl.error(f"API rejection error: {response.text}")
                except Exception as e:
                    strl.error(f"Execution pipeline structural failure: {str(e)}")

# --- MODE 2: CONVERSATIONAL CHAT APPLICATION ---
elif app_mode == "Conversational Chat Pipeline":
    strl.title("💬 Context-Isolated RAG Chat Terminal")
    strl.write("Query the local vector cluster space. The response generation is locked strictly to ingested document records.")

    # Initialize local streamlit chat state log array if missing
    if "chat_history" not in strl.session_state:
        strl.session_state.chat_history = []

    # Display active historical chat frames
    for message in strl.session_state.chat_history:
        with strl.chat_message(message["role"]):
            strl.markdown(message["content"])

    # Intercept user interaction prompt
    if user_prompt := strl.chat_input("Ask a question based on database records..."):
        # Append and display user input instant frame
        strl.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with strl.chat_message("user"):
            strl.markdown(user_prompt)
            
        # Dispatch query matrix coordinates to the API engine
        with strl.chat_message("assistant"):
            response_box = strl.empty()
            with strl.spinner("[*] Executing spatial distance lookup vs. pgvector..."):
                try:
                    res = requests.post(f"{BACKEND_API}/query", json={"prompt": user_prompt})
                    if res.status_code == 200:
                        model_output = res.json()["answer"]
                        response_box.markdown(model_output)
                        # Append assistant thread payload back to history state
                        strl.session_state.chat_history.append({"role": "assistant", "content": model_output})
                    else:
                        response_box.markdown(f"Error executing API fetch pass: {res.text}")
                except Exception as e:
                    response_box.markdown(f"API connection failure: {str(e)}")
