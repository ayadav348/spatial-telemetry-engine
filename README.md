# Spatial Telemetry Engine

A fully local Retrieval-Augmented Generation (RAG) platform for ingesting, indexing, and querying structured 3D autonomous driving telemetry. The engine stores spatial scene frames as dense vector embeddings in PostgreSQL and enables natural-language semantic search over those scenes, optionally synthesizing structured state-vector outputs via a local LLM.

---

## What It Does

The Spatial Telemetry Engine solves the problem of searching through large collections of 3D scene logs using natural language. Instead of writing SQL queries over raw coordinate data, you describe a scenario in plain English (e.g., *"find an oncoming sedan with partial occlusion under 20 meters"*) and the engine retrieves the most semantically similar frames from your indexed database.

**Core capabilities:**

1. **Spatial Frame Ingestion** — Accepts structured 3D scene frames containing ego-vehicle state, camera extrinsics, and per-object 3D bounding boxes (position, size, velocity, occlusion state). Each frame is serialized into a descriptive text string, embedded via `sentence-transformers`, and stored in PostgreSQL with the `pgvector` extension.

2. **Dataset Bridge** — Normalizes raw telemetry from heterogeneous autonomous driving datasets (nuScenes quaternion-based, Waymo yaw-based) into the engine's unified coordinate format. Handles ego-relative coordinate translation and quaternion-to-yaw conversion automatically.

3. **Semantic Scenario Search** — Embeds a natural language query, performs cosine similarity search against all indexed frames, and returns the top 3 closest matching scenes. Optionally passes the retrieved context to a local `llama3.2` instance to synthesize a structured physics summary and extract a state-space seed vector $x = [X, Y, \dot{X}, \dot{Y}]^T$ for downstream tracking filters.

4. **Streamlit Dashboard** — A browser-based UI with two panes: an ingestion workspace (with tabs for native JSON payloads and dataset bridge mode) and a conversational query workspace.

---

## Use Cases

- **Autonomous vehicle research** — Index Waymo or nuScenes log segments and retrieve frames matching specific traffic scenarios for analysis or filter initialization.
- **Tracking filter bootstrapping** — Use the query pipeline to automatically seed Kalman filter or other state estimators with real-world position and velocity vectors.
- **Dataset exploration** — Search through thousands of indexed frames without writing SQL, using scenario descriptions as queries.
- **Custom sensor data** — Ingest any 3D bounding-box telemetry source (e.g., GoPro + IPM pipeline on a custom vehicle) via the native stream endpoint.
- **Benchmarking & stress testing** — The included `benchmark_pipeline.py` generates synthetic Waymo-like sequences and measures ingestion latency and query accuracy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User Interface Layer                   │
│              Streamlit  (ui.py :8501)                   │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP POST
┌────────────────────────▼────────────────────────────────┐
│                  REST API Backend                       │
│              FastAPI + Uvicorn  (main.py :8000)         │
│                                                         │
│  POST /ingest/spatial        → native frame ingest      │
│  POST /ingest/dataset-bridge → dataset normalization    │
│  POST /query/scenario        → semantic search + LLM    │
└──────────┬─────────────────────────────┬────────────────┘
           │ psycopg2                    │ ollama (optional)
┌──────────▼──────────┐      ┌──────────▼──────────────┐
│  PostgreSQL          │      │  Ollama Daemon           │
│  + pgvector          │      │  llama3.2 (query synth.) │
│  spatial_vector_db   │      └─────────────────────────┘
└─────────────────────┘

Embedding: sentence-transformers all-MiniLM-L6-v2 (local, no daemon required)
```

| Component | File | Role |
|-----------|------|------|
| FastAPI backend | `main.py` | REST endpoints, background ingestion, vector search, LLM query |
| Streamlit UI | `ui.py` | Ingestion workspace, dataset bridge UI, conversational search |
| Dataset bridge library | `dataset_bridge.py` | nuScenes/Waymo coordinate normalization utilities |
| Benchmark suite | `benchmark_pipeline.py` | Synthetic load generation and latency profiling |
| Simulation test | `test_ingest.py` | 10-frame highway simulation ingest test |
| Waymo unit test | `test_waymo_ingest.py` | Single Waymo frame dispatch and response validation |

---

## Data Model

The engine stores one row per scene frame in PostgreSQL:

| Column | Type | Description |
|--------|------|-------------|
| `id` | serial | Auto-increment primary key |
| `scene_id` | text | Identifier for the scene or log segment |
| `frame_timestamp` | real | Timestamp offset within the sequence (seconds) |
| `ego_velocity_vector` | real[] | Ego vehicle velocity `[Vx, Vy, Vz]` in m/s |
| `raw_telemetry_json` | jsonb | Full original payload including all detected objects |
| `spatial_geometry_embedding` | vector(384) | Dense embedding of the serialized scene description |

---

## Requirements

- Python 3.10+
- PostgreSQL running locally on port `5432` with the `pgvector` extension
- Ollama (only required for the `/query/scenario` LLM synthesis step — ingestion works without it)

---

## Setup

### 1. PostgreSQL

```sql
CREATE DATABASE spatial_vector_db;
\c spatial_vector_db
CREATE EXTENSION IF NOT EXISTS vector;
```

The `spatial_scene_store` table is created automatically on first server start.

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install sentence-transformers
```

### 3. Ollama (optional — for LLM query synthesis)

```bash
ollama pull llama3.2
```

The embedding model (`all-MiniLM-L6-v2`) runs via `sentence-transformers` directly and does **not** require Ollama.

---

## Running

### Backend API

```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

Server binds to `http://127.0.0.1:8000`. Interactive API docs available at `http://127.0.0.1:8000/docs`.

### Streamlit UI

```bash
streamlit run ui.py
```

Opens at `http://localhost:8501`.

---

## API Reference

### `POST /ingest/spatial`

Ingest a native 3D scene frame. Processing is queued as a background task; the endpoint returns `202 Accepted` immediately.

**Request body:**
```json
{
  "scene_id": "miata-sequence-042",
  "frame_timestamp": 12.450,
  "ego_velocity_vector": [11.2, 0.0, -0.05],
  "camera_extrinsics_rt": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.2],
    [0.0, 0.0, 1.0, -0.4]
  ],
  "detected_objects": [
    {
      "target_id": 105,
      "classification": "car",
      "center_xyz": [2.5, 14.2, -0.1],
      "extent_lwh": [4.5, 1.8, 1.4],
      "velocity_vector": [-8.5, 0.2, 0.0],
      "occlusion_state": "partial_500ms"
    }
  ]
}
```

`center_xyz` and `velocity_vector` are relative to the ego vehicle origin. `occlusion_state` accepts `"none"`, `"partial_500ms"`, or `"total"`.

---

### `POST /ingest/dataset-bridge`

Accepts raw coordinates from nuScenes or Waymo datasets in their native formats and normalizes them before ingestion.

**nuScenes example** (quaternion orientation):
```json
{
  "source_dataset": "nuscenes",
  "scene_id": "nuscenes-log-segment-081",
  "frame_timestamp": 44.102,
  "ego_translation": [1042.4, 892.1, 12.3],
  "ego_velocity_vector": [8.4, 1.2, 0.0],
  "target_classification": "truck",
  "target_id": 4022,
  "box_translation": [1054.1, 896.7, 12.5],
  "box_size_lwh": [6.5, 2.2, 3.1],
  "orientation_parameter": [0.707, 0.0, 0.0, 0.707],
  "target_velocity_vector": [-2.1, 0.5, 0.0],
  "occlusion_state": "none"
}
```

**Waymo example** (scalar yaw heading):
```json
{
  "source_dataset": "waymo",
  "scene_id": "waymo-run-segment-119",
  "frame_timestamp": 182.904,
  "ego_translation": [45.2, -12.1, 0.5],
  "ego_velocity_vector": [14.1, 0.0, -0.02],
  "target_classification": "motorcycle",
  "target_id": 771,
  "box_translation": [52.9, -10.3, 0.4],
  "box_size_lwh": [2.1, 0.8, 1.2],
  "orientation_parameter": [1.5708],
  "target_velocity_vector": [12.4, -0.2, 0.0],
  "occlusion_state": "total"
}
```

The bridge computes ego-relative coordinates (`box_translation - ego_translation`) before storing.

---

### `POST /query/scenario`

Perform semantic search over indexed frames. Embeds the prompt, finds the 3 nearest frames by cosine distance, and (if Ollama is running) passes them to `llama3.2` to generate a structured output.

**Request:**
```json
{ "prompt": "Find an oncoming vehicle with partial occlusion closer than 15 meters" }
```

**Response:**
```json
{
  "answer": "$$x = [2.5, 14.2, -8.5, 0.2]^T$$\n- Target ID: 105\n...",
  "state_vector_seed": [2.5, 14.2, -8.5, 0.2],
  "status": "SUCCESS"
}
```

`state_vector_seed` is `[X, Y, Vx, Vy]` extracted from the LLM output, ready to initialize a tracking filter. Returns `null` if extraction fails or Ollama is not available.

---

## Ingestion Pipeline Flow

```
User submits JSON payload
        │
        ▼
  /ingest/spatial  ──────────────────────────┐
        │                                    │
        │  /ingest/dataset-bridge            │
        │    → subtract ego_translation      │
        │    → normalize to SpatialFrame     │
        │                                    │
        ▼                                    │
  Background task ◄───────────────────────────┘
        │
        ├─ serialize_spatial_frame()
        │    → text description of scene
        │
        ├─ SentenceTransformer.encode()
        │    → 384-dim float vector
        │
        └─ INSERT into spatial_scene_store
             (scene_id, timestamp, ego_vel, raw_json, embedding)
```

---

## Testing & Benchmarking

**Simulate a 10-frame highway scenario:**
```bash
python test_ingest.py
```

**Send a single Waymo frame:**
```bash
python test_waymo_ingest.py
```

**Full stress test (150 frames + query latency profiling):**
```bash
python benchmark_pipeline.py
```

The benchmark generates a synthetic 150-frame Waymo SF Mission sequence at 10 Hz, measures per-frame HTTP ingestion latency (avg, P99, peak), then executes a semantic query and prints the extracted state vector.

---

## Notes

- The `/query/scenario` endpoint requires a running Ollama instance with `llama3.2` pulled. Ingestion (`/ingest/spatial`, `/ingest/dataset-bridge`) works entirely without Ollama.
- The embedding model (`all-MiniLM-L6-v2`, 384 dimensions) is downloaded automatically by `sentence-transformers` on first run.
- DB credentials are hardcoded in `main.py` (`DB_PARAMS`). Update to match your local PostgreSQL configuration.
- The `dataset_bridge.py` module contains a standalone `DatasetTelemetryBridge` class with `extract_nuscenes_kinematics` and `extract_waymo_kinematics` methods that can be used independently of the API.
