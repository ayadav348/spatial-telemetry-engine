# Spatial Telemetry Engine

A fully local RAG + NL-to-SQL platform for ingesting, indexing, and querying structured 3D autonomous driving telemetry. The engine stores spatial scene frames as dense vector embeddings in PostgreSQL and supports two complementary query modes: natural-language semantic search over scene embeddings, and schema-aware natural-language SQL generation executed directly against the telemetry store.

---

## What It Does

The Spatial Telemetry Engine solves two distinct problems over 3D AV scene logs:

1. **Semantic scenario search** — Describe a traffic scenario in plain English (e.g., *"find an oncoming sedan with partial occlusion under 20 meters"*) and retrieve the most semantically similar indexed frames via cosine similarity over dense vector embeddings.

2. **Natural language SQL** — Ask structured questions in plain English (e.g., *"show frames where the ego vehicle was traveling faster than 10 m/s"*) and have the engine automatically generate, validate, and execute a safe PostgreSQL query against the live telemetry store — no SQL knowledge required.

**Core capabilities:**

1. **Spatial Frame Ingestion** — Accepts structured 3D scene frames containing ego-vehicle state, camera extrinsics, and per-object 3D bounding boxes (position, size, velocity, occlusion state). Each frame is serialized into a descriptive text string, embedded via `sentence-transformers`, and stored in PostgreSQL with the `pgvector` extension.

2. **Dataset Bridge** — Normalizes raw telemetry from heterogeneous autonomous driving datasets (nuScenes quaternion-based, Waymo yaw-based) into the engine's unified coordinate format. Handles ego-relative coordinate translation and quaternion-to-yaw conversion automatically.

3. **Semantic Scenario Search** — Embeds a natural language query, performs cosine similarity search against all indexed frames, and returns the closest matching scenes. Optionally passes the retrieved context to a local `llama3.2` instance to synthesize a structured physics summary and extract a state-space seed vector $x = [X, Y, \dot{X}, \dot{Y}]^T$ for downstream tracking filters.

4. **NL-to-SQL Query Engine** — Introspects the live PostgreSQL schema at query time, injects it as structured context into a `llama3.2` prompt with AV-domain annotations, extracts and validates the generated SQL (SELECT-only, injection-safe), executes it on a read-only connection, and returns raw results as a structured JSON response.

5. **Streamlit Dashboard** — A browser-based UI with an ingestion workspace (native JSON and dataset bridge tabs) and a query workspace (semantic search tab and NL-to-SQL tab with preset queries and live results table).

6. **Manual Database Control** — Data persists across server restarts. A dedicated `DELETE /db/clear` endpoint (with a UI button) allows on-demand wipe of all frames for re-ingestion demos.

---

## Use Cases

- **Autonomous vehicle research** — Index Waymo or nuScenes log segments and retrieve frames matching specific traffic scenarios for analysis or filter initialization.
- **Tracking filter bootstrapping** — Use the semantic query pipeline to automatically seed Kalman filter or other state estimators with real-world position and velocity vectors.
- **Dataset exploration** — Search through indexed frames without writing SQL, using either scenario descriptions (vector search) or structured questions (NL-to-SQL).
- **Custom sensor data** — Ingest any 3D bounding-box telemetry source (e.g., GoPro + IPM pipeline on a custom vehicle) via the native stream endpoint.
- **Benchmarking & stress testing** — The included `benchmark_pipeline.py` generates synthetic Waymo-like sequences and measures ingestion latency and query accuracy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│                Streamlit  (ui.py :8501)                     │
│                                                             │
│  Ingestion Workspace        Query Workspace                 │
│  ├─ Native Stream Payload   ├─ NL → SQL Query (new)        │
│  └─ Dataset Bridge          └─ Scenario Vector Search       │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP
┌────────────────────────▼────────────────────────────────────┐
│                    REST API Backend                         │
│                FastAPI + Uvicorn  (main.py :8000)           │
│                                                             │
│  POST   /ingest/spatial        → native frame ingest        │
│  POST   /ingest/dataset-bridge → dataset normalization      │
│  POST   /query/scenario        → semantic search + LLM      │
│  POST   /query/sql             → NL-to-SQL generation       │
│  GET    /db/stats              → frame count                │
│  DELETE /db/clear              → manual database wipe       │
└──────────┬──────────────────────────────┬───────────────────┘
           │ psycopg2                     │ ollama
┌──────────▼──────────┐       ┌──────────▼──────────────────┐
│  PostgreSQL          │       │  Ollama Daemon               │
│  + pgvector          │       │  llama3.2                    │
│  spatial_vector_db   │       │  ├─ scenario synthesis       │
└─────────────────────┘       │  └─ SQL generation           │
                               └─────────────────────────────┘

Embedding: sentence-transformers all-MiniLM-L6-v2 (local, no daemon required)
```

| Component | File | Role |
|-----------|------|------|
| FastAPI backend | `main.py` | REST endpoints, ingestion, vector search, NL-to-SQL, DB management |
| Streamlit UI | `ui.py` | Ingestion workspace, dataset bridge, semantic search, SQL query tab |
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

Data **persists across server restarts**. Use `DELETE /db/clear` or the UI button to wipe frames manually.

---

## Requirements

- Python 3.10+
- PostgreSQL running locally on port `5432` with the `pgvector` extension
- Ollama with `llama3.2` pulled (required for `/query/scenario` LLM synthesis and `/query/sql` generation — ingestion works without it)

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

### 3. Ollama (required for both query endpoints)

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

Semantic search over indexed frames. Embeds the prompt, finds nearest frames by cosine distance, and passes retrieved context to `llama3.2` to synthesize a structured output.

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

### `POST /query/sql`

Schema-aware natural language to SQL. Introspects the live database schema, injects it with AV-domain context into a `llama3.2` prompt, validates the generated query (SELECT-only, blocklist enforcement, table allowlist), and executes it on a read-only connection.

**Request:**
```json
{ "prompt": "Show frames where the ego vehicle was traveling faster than 10 m/s" }
```

**Response:**
```json
{
  "generated_sql": "SELECT id, scene_id, frame_timestamp, ego_velocity_vector\nFROM spatial_scene_store\nWHERE sqrt(ego_velocity_vector[1]^2 + ego_velocity_vector[2]^2 + ego_velocity_vector[3]^2) > 10.0\nORDER BY frame_timestamp\nLIMIT 50;",
  "rows": [
    {
      "id": 2,
      "scene_id": "waymo-sf-mission-seq-1092",
      "frame_timestamp": 182.904,
      "ego_velocity_vector": [14.15, 0.22, -0.02]
    }
  ],
  "row_count": 1,
  "status": "SUCCESS"
}
```

Status values: `SUCCESS`, `VALIDATION_FAILED` (with `detail` explaining rejection reason), `EXECUTION_ERROR` (with `detail` containing the PostgreSQL error).

Security model:
- Only `SELECT` statements are permitted
- Blocklist rejects: `DROP`, `DELETE`, `UPDATE`, `INSERT`, `TRUNCATE`, `ALTER`, `CREATE`, `EXECUTE`, `GRANT`, `REVOKE`, `--`, `/*`
- Only `spatial_scene_store` may be referenced — cross-table queries are rejected
- Execution runs on a `readonly=True` psycopg2 connection as a PostgreSQL-level safeguard
- The `spatial_geometry_embedding` vector column is excluded from results automatically

---

### `GET /db/stats`

Returns the current frame count.

**Response:** `{ "frame_count": 2 }`

---

### `DELETE /db/clear`

Wipes all telemetry frames and resets the identity sequence. Equivalent to `TRUNCATE TABLE spatial_scene_store RESTART IDENTITY`.

**Response:** `{ "status": "CLEARED", "message": "All telemetry frames purged. Identity sequence reset." }`

---

## NL-to-SQL Pipeline Flow

```
User natural language prompt
          │
          ▼
extract_schema_context()
  → queries information_schema.columns
  → annotates each column with AV domain meaning
  → documents JSONB structure of raw_telemetry_json
          │
          ▼
llama3.2 (via Ollama)
  → system prompt: schema context + AV domain rules + output format rules
  → user message: natural language prompt
          │
          ▼
extract_sql_block()
  → strips ```sql ... ``` fences from LLM output
          │
          ▼
validate_generated_sql()
  → must start with SELECT
  → blocklist check (DROP, DELETE, INSERT, etc.)
  → table allowlist (spatial_scene_store only)
          │
          ├─ VALIDATION_FAILED → return early with rejection reason
          │
          ▼
psycopg2 (readonly=True connection)
  → cursor.execute(generated_sql)
          │
          ▼
Result serialization
  → embedding column replaced with "[vector omitted]"
  → numpy types converted to Python native
          │
          ▼
{ generated_sql, rows, row_count, status }
```

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

- Both `/query/scenario` and `/query/sql` require a running Ollama instance with `llama3.2` pulled. Ingestion works entirely without Ollama.
- The embedding model (`all-MiniLM-L6-v2`, 384 dimensions) is downloaded automatically by `sentence-transformers` on first run.
- DB credentials are hardcoded in `main.py` (`DB_PARAMS`). Update to match your local PostgreSQL configuration.
- The `dataset_bridge.py` module contains a standalone `DatasetTelemetryBridge` class with `extract_nuscenes_kinematics` and `extract_waymo_kinematics` methods that can be used independently of the API.
- Data persists across server restarts. Use the sidebar "Clear Database" button in the UI or `DELETE /db/clear` directly to wipe frames between demo runs.
