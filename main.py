import os
import numpy as np
import ollama
from pypdf import PdfReader

class LocalRAGPipeline:
    def __init__(self, llm_model="llama3.2", embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=100):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []        # Holds text content blocks
        self.embeddings = []       # Holds raw continuous vector coordinate matrices

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Parses a native digital PDF and extracts raw string data."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Target PDF not found at: {pdf_path}")

        reader = PdfReader(pdf_path)
        extracted_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text)

    def chunk_text(self, text: str):
        """Slices massive string streams into overlapping tokens to preserve context boundaries."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest_data(self, source_type: str, payload: str):
        """Processes raw input data (either text or a path to a PDF), generates embeddings

        via the dedicated embedding model, and stores them in a memory-mapped numpy array.
        """
        print(f"[*] Initializing compilation pass for input source...")
        raw_text = ""

        if source_type.lower() == "pdf":
            raw_text = self.extract_pdf_text(payload)
        else:
            raw_text = payload

        # Chunk the payload
        chunks = self.chunk_text(raw_text)
        self.documents = chunks

        # Batch generation of semantic embeddings via Ollama's dedicated embedding endpoint
        print(f"[*] Compiling vector space matrices for {len(chunks)} document chunks using {self.embedding_model}...")
        for chunk in chunks:
            response = ollama.embed(model=self.embedding_model, input=chunk)
            # Unpack payload variant across client versions
            vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']
            self.embeddings.append(vector)

        # Convert internal list tracking to an efficient numpy matrix array
        self.embeddings = np.array(self.embeddings)
        print(f"[+] Ingestion vector compilation successful. Pipeline locked and loaded.")

    def _cosine_similarity(self, vec_a, vec_b):
        """Computes the dot product normalized by the L2 norm of the vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def query(self, user_prompt: str, top_k: int = 3) -> str:
        """Executes similarity verification against vectors, constructs context,

        and streams response from the generative chat model.
        """
        if len(self.documents) == 0:
            return "Pipeline Error: Vector database target is empty. Run ingest_data first."

        # 1. Compile incoming user prompt into a coordinate using the SAME embedding matrix
        prompt_res = ollama.embed(model=self.embedding_model, input=user_prompt)
        prompt_vector = prompt_res['embeddings'][0] if 'embeddings' in prompt_res else prompt_res['embedding']
        prompt_vector = np.array(prompt_vector)

        # 2. Iterate through vector matrices to determine highest spatial cosine similarity
        scores = []
        for doc_vector in self.embeddings:
            score = self._cosine_similarity(prompt_vector, doc_vector)
            scores.append(score)

        # 3. Pull indices of top_k absolute matches
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Assemble reference context block
        context_blocks = [self.documents[idx] for idx in top_indices]
        context_payload = "\n---\n".join(context_blocks)

        # 4. Construct system context payload to isolate prompt leakage
        system_instructions = (
            "You are an advanced technical intelligence assistant. You are given the following context blocks "
            "extracted from system documentation. Synthesize an optimal answer to the user's prompt strictly "
            "utilizing this verified context layer. If the context does not contain the information required, "
            "state that clearly.\n\n"
            f"=== VERIFIED CONTEXT BLOCK ===\n{context_payload}\n==============================="
        )

        # 5. Stream the model inference pass directly to stdout via the generative LLM
        response_stream = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )

        print(f"\n[Aarav-Inference-Engine]: ", end="")
        full_response = ""
        for chunk in response_stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_response += content
        print("\n")
        return full_response


# --- Execution Controller Execution Example ---
if __name__ == "__main__":
    # Ensure you ran: ollama pull nomic-embed-text
    # Ensure you ran: ollama pull llama3.2
    pipeline = LocalRAGPipeline(llm_model="llama3.2", embedding_model="nomic-embed-text")

    print("=== Local RAG Pipeline Protocol Initialized ===")
    choice = input("Select input source track - [1] Raw Text | [2] Local PDF File: ").strip()

    if choice == "1":
        text_input = input("\nPaste raw text data block: ")
        pipeline.ingest_data(source_type="text", payload=text_input)
    elif choice == "2":
        pdf_path = input("\nEnter exact system path to target PDF: ").strip()
        pipeline.ingest_data(source_type="pdf", payload=pdf_path)
    else:
        print("Invalid termination signal.")
        exit()

    # Enter conversational processing loop
    print("\n--- Pipeline Active: Enter prompts or type 'exit' to terminate thread ---")
    while True:
        user_query = input("\nUser Prompt >> ")
        if user_query.lower() in ['exit', 'quit']:
            print("Disconnecting active tracking matrices.")
            break
        if not user_query.strip():
            continue

        pipeline.query(user_prompt=user_query, top_k=2)
