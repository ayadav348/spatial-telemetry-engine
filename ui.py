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
    strl.write("Upload a PDF document or paste raw text to compile context into the local `pgvector` store.")

    # Two-column layout: PDF upload on the left, text input on the right
    col_pdf, col_text = strl.columns(2)

    with col_pdf:
        strl.subheader("Upload PDF Document")
        uploaded_file = strl.file_uploader("Upload Target PDF Document", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            strl.success(f"File loaded: {uploaded_file.name}")

    with col_text:
        strl.subheader("Paste Raw Text")
        manual_text = strl.text_area(
            "Paste context text here",
            height=200,
            placeholder="Paste textbook passages, notes, or any reference material here...",
            label_visibility="collapsed"
        )
        manual_text_name = strl.text_input(
            "Label for this text (used as filename in the database)",
            placeholder="e.g. chapter1_notes",
            value=""
        )

    strl.divider()

    if strl.button("Compile & Generate Embeddings", type="primary"):
        sources_queued = []

        # Collect PDF source
        if uploaded_file is not None:
            sources_queued.append(("pdf", uploaded_file))

        # Collect manual text source
        if manual_text.strip():
            label = manual_text_name.strip() if manual_text_name.strip() else "manual_text_input"
            sources_queued.append(("text", (label, manual_text)))

        if not sources_queued:
            strl.warning("No input provided. Upload a PDF or paste text before compiling.")
        else:
            for source_type, source_data in sources_queued:
                if source_type == "pdf":
                    with strl.spinner(f"[*] Extracting text from {source_data.name}..."):
                        try:
                            # Extract text from uploaded PDF memory stream
                            reader = PdfReader(source_data)
                            extracted_text = []
                            for page in reader.pages:
                                text = page.extract_text()
                                if text:
                                    extracted_text.append(text)

                            full_raw_text = "\n".join(extracted_text)

                            if not full_raw_text.strip():
                                strl.error(f"Extraction error for {source_data.name}: File contains zero token sequences or is an un-OCRed image scan.")
                            else:
                                payload = {"filename": source_data.name, "text_content": full_raw_text}
                                response = requests.post(f"{BACKEND_API}/ingest", json=payload)
                                if response.status_code == 200:
                                    data = response.json()
                                    strl.success(f"[+] PDF '{source_data.name}' ingested. Wrote {data['chunks_ingested']} context blocks to pgvector.")
                                else:
                                    strl.error(f"API rejection error for PDF: {response.text}")
                        except Exception as e:
                            strl.error(f"PDF pipeline failure: {str(e)}")

                elif source_type == "text":
                    label, raw_text = source_data
                    with strl.spinner(f"[*] Ingesting text block '{label}'..."):
                        try:
                            payload = {"filename": label, "text_content": raw_text}
                            response = requests.post(f"{BACKEND_API}/ingest", json=payload)
                            if response.status_code == 200:
                                data = response.json()
                                strl.success(f"[+] Text '{label}' ingested. Wrote {data['chunks_ingested']} context blocks to pgvector.")
                            else:
                                strl.error(f"API rejection error for text: {response.text}")
                        except Exception as e:
                            strl.error(f"Text pipeline failure: {str(e)}")

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
