#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  fill: rgb("#fafafa")
)
#set text(
  font: "Liberation Serif",
  size: 10.5pt,
  fill: rgb("#1a1a1a")
)

// Header Section
#block(
  width: 100%,
  stroke: (bottom: 1pt + rgb("#cbd5e1")),
  inset: (bottom: 35pt),
  [
    #text(weight: "bold", size: 18pt, fill: rgb("#0f172a"))[Technical Brief: Local Vector Architecture & RAG Pipelines]     #v(2pt)
    #text(size: 10pt, fill: rgb("#64748b"))[Systems Optimization & Semantic Coordinate Routing]
  ]
)

== 1. Mathematical Core: Embedding Models
An embedding model turns human language into high-dimensional vector coordinates $bb(R)^d$. Unlike keyword-based parsers that match exact text, an embedding network uses self-attention to place concepts into spatial regions where meaning determines distance.

Given an input string, the model outputs a normalized floating-point array with a fixed dimension ($d = 768$ for `nomic-embed-text`). Relationships between vectors are computed using *Cosine Similarity*, which measures the angle between two vectors:

$"Similarity" = cos(theta) = (bb(A) dot bb(B)) / (||bb(A)|| ||bb(B)||)$

When $theta approx 0$, the vectors point in the same direction, which means their meanings are close even if they use different words.

== 2. Local Storage Layer: pgvector
Standard relational databases struggle with high-dimensional nearest-neighbor searches. `pgvector` extends PostgreSQL to natively store, index, and query these vector arrays using special operators.

It supports Approximate Nearest Neighbor (ANN) searches with *HNSW (Hierarchical Navigable Small World)* graphs or *IVFFlat* inverted files, avoiding slow $O(N)$ full table scans. Instead of matching strings, the database runs spatial distance calculations directly, finding matching rows using the cosine distance operator (`<=>`).

== 3. Architecture of the Local RAG Pipeline
Retrieval-Augmented Generation (RAG) reduces hallucinations by injecting verified information into the model's context window.

#block(
  fill: rgb("#f1f5f9"),
  inset: 10pt,
  radius: 4pt,
  stroke: 0.5pt + rgb("#e2e8f0"),
  [
    #text(weight: "bold", size: 9.5pt, fill: rgb("#334155"))[Pipeline Steps:]
    + *Data Ingestion:* Text or PDF files are read into strings.
    + *Context Chunking:* Text is split using a sliding window (`chunk_size=500`, `chunk_overlap=100`) so context carries across boundaries.
    + *Spatial Encoding:* Chunks are sent to `nomic-embed-text` via Ollama to generate vector coordinates, then stacked in memory with NumPy.
    + *Query Mapping & Distance Scan:* The input query is converted to a vector. A distance search finds the top-$k$ matching chunks.
    + *Context Isolation & Chat Inference:* The matched chunks are placed into a system prompt and sent to `llama3.2` to stream a response.
  ]
)

== 4. Ecosystem Synergy Matrix
The components work together in an integrated data processing loop:

#table(
  columns: (1fr, 2fr, 1.5fr),
  fill: (x, y) => if y == 0 { rgb("#1e293b") } else if calc.even(y) { rgb("#f8fafc") } else { rgb("#ffffff") },
  stroke: 0.5pt + rgb("#e2e8f0"),
  inset: 7pt,
  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },
  
  [#text(fill: white, weight: "bold")[Component]], [#text(fill: white, weight: "bold")[What It Does]], [#text(fill: white, weight: "bold")[Pipeline Role]],
  [#text(fill: white)[Ollama API]], [Manages local model endpoints and handles concurrency for the encoder and generative model.], [System Execution Interface],
  [Nomic Embed], [Converts text into fixed-width vector coordinates ($bb(R)^768$).], [Deterministic Math Compiler],
  [NumPy Matrix / pgvector], [Stores raw vectors and runs fast distance calculations using linear algebra.], [Spatial Storage & Search Engine],
  [Llama 3.2], [Processes structured context and generates human-readable responses.], [Inference Generation Layer]
)

== 5. Python Implementation
The project includes a `LocalRAGPipeline` class in Python that implements this architecture. It ingests text or PDF files, splits them into overlapping chunks, generates embeddings locally via Ollama's `nomic-embed-text`, and stores them in a NumPy matrix. On query, it converts the prompt to a vector, finds the top-$k$ most similar chunks by cosine similarity, and sends them as context to `llama3.2` for generation.
