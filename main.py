import os
import subprocess
import time
import socket
import json
import math
import re
import functools
import threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg2
from psycopg2 import pool as psycopg2_pool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI(title="Spatial Telemetry Engine", description="Local 3D Scene Discovery & Volumetric Retrieval Platform")

DB_PARAMS = {
    "dbname": "spatial_vector_db",
    "user": "postgres",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

# --- Connection Pool Layer ---
# Replaces per-request psycopg2.connect() (new TCP handshake + session setup every call)
# with a reusable pool of warm connections. This is the dominant latency win for the
# NL->SQL pipeline under repeated queries.
CONNECTION_POOL: Optional[psycopg2_pool.ThreadedConnectionPool] = None
_POOL_LOCK = threading.Lock()
# Track which physical connections have had the pgvector type adapter registered,
# so we only register once per connection rather than on every checkout.
_VECTOR_REGISTERED_CONNS = set()


def init_pool():
    """Lazily build the global threaded connection pool. Safe to call repeatedly."""
    global CONNECTION_POOL
    if CONNECTION_POOL is None:
        with _POOL_LOCK:
            if CONNECTION_POOL is None:
                CONNECTION_POOL = psycopg2_pool.ThreadedConnectionPool(
                    minconn=2, maxconn=10, **DB_PARAMS
                )


def get_conn(register_vec: bool = False):
    """Check out a connection from the pool. Optionally register the pgvector adapter."""
    if CONNECTION_POOL is None:
        init_pool()
    conn = CONNECTION_POOL.getconn()
    if register_vec and id(conn) not in _VECTOR_REGISTERED_CONNS:
        register_vector(conn)
        _VECTOR_REGISTERED_CONNS.add(id(conn))
    return conn


def put_conn(conn):
    """Return a connection to the pool. Reset session state so readonly does not leak."""
    if conn is None or CONNECTION_POOL is None:
        return
    try:
        # Roll back any open transaction and clear readonly flag before reuse.
        conn.rollback()
        conn.set_session(readonly=False)
    except Exception:
        pass
    CONNECTION_POOL.putconn(conn)

# High-Velocity Native Local Embedding Layer
SPATIAL_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def get_spatial_embedding(text_payload: str):
    """Generates dense vector matrices instantly on local hardware."""
    vector_array = SPATIAL_EMBEDDER.encode(text_payload)
    return vector_array.tolist()

def init_db():
    """Initializes extension registers and spatial telemetry store tables."""
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spatial_scene_store (
                id serial PRIMARY KEY,
                scene_id text NOT NULL,
                frame_timestamp real NOT NULL,
                ego_velocity_vector real[] NOT NULL,
                raw_telemetry_json jsonb NOT NULL,
                spatial_geometry_embedding vector(384)
            );
        """)
        conn.commit()

        # --- Vector Index ---
        # Without an ANN index, every <=> cosine-distance query performs a full
        # sequential scan. HNSW reduces vector search from O(n) to ~O(log n).
        # Requires pgvector >= 0.5.0; degrade gracefully on older builds.
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spatial_embedding_hnsw
                ON spatial_scene_store
                USING hnsw (spatial_geometry_embedding vector_cosine_ops);
            """)
            conn.commit()
            print("[Database Initializer] HNSW vector index ready.")
        except Exception as idx_err:
            conn.rollback()
            print(f"[Database Initializer] HNSW index unavailable ({idx_err}); attempting IVFFlat fallback.")
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_spatial_embedding_ivf
                    ON spatial_scene_store
                    USING ivfflat (spatial_geometry_embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                conn.commit()
                print("[Database Initializer] IVFFlat vector index ready.")
            except Exception as ivf_err:
                conn.rollback()
                print(f"[Database Initializer] No vector index created ({ivf_err}); sequential scan in use.")

        cursor.close()
    finally:
        put_conn(conn)

@app.on_event("startup")
def startup_event():
    init_pool()
    init_db()
    print("[Database Initializer] Spatial telemetry store initialized. Existing frames preserved.")

# --- Metrics Verification Endpoint ---
@app.get("/db/stats")
def get_db_stats():
    """Returns database telemetry frame counts to verify initialization wiping."""
    conn = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM spatial_scene_store;")
        count = cursor.fetchone()[0]
        cursor.close()
        return {"frame_count": count}
    except Exception as e:
        return {"frame_count": -1, "error": str(e)}
    finally:
        put_conn(conn)

# --- Manual Database Clear Endpoint ---
@app.delete("/db/clear")
def clear_database():
    """Wipes all telemetry frames from the store. Resets identity sequence."""
    conn = None
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE spatial_scene_store RESTART IDENTITY;")
            conn.commit()
        return {"status": "CLEARED", "message": "All telemetry frames purged. Identity sequence reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Clear Failed: {str(e)}")
    finally:
        put_conn(conn)

# --- Pydantic Validation Models ---

class BoundingBox3D(BaseModel):
    target_id: int
    classification: str
    center_xyz: List[float] = Field(..., min_length=3, max_length=3)
    extent_lwh: List[float] = Field(..., min_length=3, max_length=3)
    velocity_vector: List[float] = Field(..., min_length=3, max_length=3)
    occlusion_state: str

class SpatialFramePayload(BaseModel):
    scene_id: str
    frame_timestamp: float
    ego_velocity_vector: List[float] = Field(..., min_length=3, max_length=3)
    camera_extrinsics_rt: List[List[float]]
    detected_objects: List[BoundingBox3D]

class DatasetBridgePayload(BaseModel):
    source_dataset: str
    scene_id: str
    frame_timestamp: float
    ego_translation: List[float]
    ego_velocity_vector: List[float]
    target_classification: str
    target_id: int
    box_translation: List[float]
    box_size_lwh: List[float]
    orientation_parameter: List[float]
    target_velocity_vector: List[float]
    occlusion_state: str

class QueryPayload(BaseModel):
    prompt: str

class SQLQueryPayload(BaseModel):
    prompt: str

def serialize_spatial_frame(payload: SpatialFramePayload) -> str:
    ego_speed = (sum(v**2 for v in payload.ego_velocity_vector))**0.5
    structural_string = (
        f"Sequence frame token: {payload.scene_id}. Physical Timestamp offsets: {payload.frame_timestamp:.3f}s. "
        f"Ego vehicle linear velocity state magnitude: {ego_speed:.2f} m/s. "
    )
    for idx, box in enumerate(payload.detected_objects):
        distance_euclidean = (sum(c**2 for c in box.center_xyz))**0.5
        structural_string += (
            f"Object index: {idx} [ID: {box.target_id}] Type: {box.classification}. "
            f"Position vector coordinates X:{box.center_xyz[0]:.2f} Y:{box.center_xyz[1]:.2f} Z:{box.center_xyz[2]:.2f}. "
            f"Radial distance to target: {distance_euclidean:.2f} meters. "
            f"Extent boundaries length-width-height: {' '.join(map(str, box.extent_lwh))}. "
            f"Trajectory velocity component vectors Vx:{box.velocity_vector[0]:.2f} Vy:{box.velocity_vector[1]:.2f}. "
            f"Occlusion status flag: {box.occlusion_state}. "
        )
    return structural_string

def process_and_store_telemetry(payload: SpatialFramePayload):
    conn = None
    try:
        serialized_context = serialize_spatial_frame(payload)
        spatial_vector = get_spatial_embedding(serialized_context)

        conn = get_conn(register_vec=True)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO spatial_scene_store
            (scene_id, frame_timestamp, ego_velocity_vector, raw_telemetry_json, spatial_geometry_embedding)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                payload.scene_id,
                payload.frame_timestamp,
                payload.ego_velocity_vector,
                json.dumps(payload.model_dump()),
                spatial_vector
            )
        )
        conn.commit()
        cursor.close()
        print(f"[+] Asynchronously indexed tracking metrics for sequence frame: {payload.scene_id}")
    except Exception as e:
        print(f"[-] Background Ingestion Thread Exception: {str(e)}")
    finally:
        put_conn(conn)

@app.post("/ingest/spatial", status_code=status.HTTP_202_ACCEPTED)
async def ingest_spatial_telemetry(payload: SpatialFramePayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_and_store_telemetry, payload)
    return {
        "status": "QUEUED",
        "scene_indexed": payload.scene_id,
        "frame_timestamp": payload.frame_timestamp,
        "detail": "Spatial extraction processing initiated on worker thread loop."
    }

@app.post("/ingest/dataset-bridge", status_code=status.HTTP_202_ACCEPTED)
async def ingest_through_dataset_bridge(payload: DatasetBridgePayload, background_tasks: BackgroundTasks):
    try:
        relative_center = [
            payload.box_translation[0] - payload.ego_translation[0],
            payload.box_translation[1] - payload.ego_translation[1],
            payload.box_translation[2] - payload.ego_translation[2]
        ]

        normalized_box = BoundingBox3D(
            target_id=payload.target_id,
            classification=payload.target_classification,
            center_xyz=relative_center,
            extent_lwh=payload.box_size_lwh,
            velocity_vector=payload.target_velocity_vector,
            occlusion_state=payload.occlusion_state
        )

        native_payload = SpatialFramePayload(
            scene_id=payload.scene_id,
            frame_timestamp=payload.frame_timestamp,
            ego_velocity_vector=payload.ego_velocity_vector,
            camera_extrinsics_rt=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            detected_objects=[normalized_box]
        )

        background_tasks.add_task(process_and_store_telemetry, native_payload)
        
        return {
            "status": "QUEUED",
            "source_dataset": payload.source_dataset,
            "scene_id": payload.scene_id,
            "detail": "Dataset bridge parsing complete. Ingestion job offloaded to background loop."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset Transformation Exception: {str(e)}")

def extract_state_vector(llm_text: str) -> Optional[List[float]]:
    """Parses strict LaTeX state-space arrays from local LLM context blocks."""
    math_pattern = r"x\s*=\s*\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]\^T"
    match = re.search(math_pattern, llm_text)
    if match:
        return [float(c) for c in match.groups()]

    try:
        x_match = re.search(r"Relative Position:\s*X:\s*\[?(-?\d+(?:\.\d+)?)\]?", llm_text)
        y_match = re.search(r"Relative Position:.*?Y:\s*\[?(-?\d+(?:\.\d+)?)\]?", llm_text)
        vx_match = re.search(r"Velocity Vectors:\s*Vx:\s*\[?(-?\d+(?:\.\d+)?)\]?", llm_text)
        vy_match = re.search(r"Velocity Vectors:.*?Vy:\s*\[?(-?\d+(?:\.\d+)?)\]?", llm_text)

        if x_match and y_match and vx_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            vx = float(vx_match.group(1))
            vy = float(vy_match.group(1)) if vy_match else 0.0
            return [x, y, vx, vy]
    except Exception as e:
        print(f"[-] Regex parameter fallback exception: {e}")
    return None

@functools.lru_cache(maxsize=1)
def extract_schema_context() -> str:
    """Introspects live PostgreSQL schema for spatial_scene_store and returns a formatted context string.

    The schema is static at runtime, so the result is cached after the first call.
    This removes a full DB round-trip from the hot path of every /query/sql request.
    """
    conn = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = 'spatial_scene_store'
            ORDER BY ordinal_position;
        """)
        rows = cursor.fetchall()
        cursor.close()

        annotations = {
            "id":                          "auto-increment primary key — do not filter on this",
            "scene_id":                    "text label for the scene or log segment (e.g. 'waymo-sf-mission-seq-1092')",
            "frame_timestamp":             "time offset in seconds (real/float)",
            "ego_velocity_vector":         "ego vehicle velocity array [Vx, Vy, Vz] in m/s — use 1-based index operators ONLY: ego_velocity_vector[1], ego_velocity_vector[2], ego_velocity_vector[3]. NEVER use ANY() in WHERE clauses.",
            "raw_telemetry_json":          "full JSONB scene payload — use -> and ->> operators to extract detected_objects, classification, occlusion_state, center_xyz, velocity_vector",
            "spatial_geometry_embedding":  "384-dim pgvector embedding — NEVER include in SELECT or WHERE clauses",
        }

        lines = ["Table: spatial_scene_store", "Columns:"]
        for col_name, data_type, udt_name in rows:
            display_type = udt_name if data_type == "USER-DEFINED" else data_type
            note = annotations.get(col_name, "")
            lines.append(f"  - {col_name} ({display_type}){': ' + note if note else ''}")

        lines += [
            "",
            "JSONB structure of raw_telemetry_json:",
            "  {",
            '    "scene_id": str,',
            '    "frame_timestamp": float,',
            '    "ego_velocity_vector": [Vx, Vy, Vz],',
            '    "camera_extrinsics_rt": [[...], [...], [...]],',
            '    "detected_objects": [',
            '      {',
            '        "target_id": int,',
            '        "classification": str,   -- e.g. "car", "truck", "motorcycle"',
            '        "center_xyz": [X, Y, Z], -- position relative to ego vehicle in meters',
            '        "extent_lwh": [L, W, H],',
            '        "velocity_vector": [Vx, Vy, Vz],',
            '        "occlusion_state": str   -- one of: "none", "partial_500ms", "total"',
            '      }',
            '    ]',
            '  }',
        ]
        return "\n".join(lines)
    except Exception as e:
        # Do not let lru_cache memoize a failure — raise so the next call retries.
        raise RuntimeError(f"Schema introspection failed: {str(e)}")
    finally:
        put_conn(conn)


def _strip_sql_comments(sql: str) -> str:
    """Removes SQL line comments (--) and block comments (/* */) from a query string."""
    # Remove block comments first
    sql = re.sub(r'/\*[\s\S]*?\*/', ' ', sql)
    # Remove line comments
    sql = re.sub(r'--[^\n]*', ' ', sql)
    return sql.strip()


def validate_generated_sql(sql: str) -> tuple:
    """Validates LLM-generated SQL. Returns (is_valid: bool, reason: str)."""
    # Strip comments before any analysis so they don't trigger false positives
    clean_sql = _strip_sql_comments(sql)
    stripped = clean_sql.strip().upper()

    if not stripped.startswith("SELECT") and not re.match(r'^\s*WITH\b', stripped):
        return False, "Only SELECT queries are permitted."

    # Use word-boundary matching to avoid false positives on column/alias names
    # that contain blocklist words as substrings (e.g. "update_count", "created_at")
    dml_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE",
        "ALTER", "CREATE", "GRANT", "REVOKE",
    ]
    for term in dml_keywords:
        if re.search(rf'\b{term}\b', stripped):
            return False, f"Forbidden keyword detected: '{term}'"

    # Block dangerous function calls that work inside a SELECT
    dangerous_funcs = ["PG_READ_FILE", "PG_LS_DIR", "LO_EXPORT", "PG_SLEEP", "EXEC(", "EXECUTE(", "XP_"]
    for term in dangerous_funcs:
        if term in stripped:
            return False, f"Forbidden function/keyword detected: '{term}'"

    # Block multi-statement injection (bare semicolon followed by another statement)
    if re.search(r';\s*\S', stripped):
        return False, "Multi-statement queries are not permitted."

    allowed_tables = {"SPATIAL_SCENE_STORE"}

    # Extract table names from FROM and JOIN clauses; handle optional schema prefix
    from_tables = re.findall(r'FROM\s+(?:\w+\.)?(\w+)', stripped)
    join_tables = re.findall(r'JOIN\s+(?:\w+\.)?(\w+)', stripped)
    referenced = set(from_tables + join_tables)

    # Remove CTE names from the referenced set — they are not real tables.
    # A CTE is defined as: WITH <name> AS (
    cte_names = set(re.findall(r'\bWITH\s+(\w+)\s+AS\s*\(', stripped))
    # Also handle comma-separated CTEs: , <name> AS (
    cte_names |= set(re.findall(r',\s*(\w+)\s+AS\s*\(', stripped))
    referenced -= cte_names

    for table in referenced:
        if table not in allowed_tables:
            return False, f"Unauthorized table reference: '{table.lower()}'"

    return True, ""


def extract_sql_block(llm_text: str) -> str:
    """Extracts a SQL query from an LLM response, stripping markdown fences if present."""
    # Try to extract from ```sql ... ``` block
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]+?)```", llm_text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    # Fallback: find first SELECT statement
    select_match = re.search(r"(SELECT[\s\S]+?;)", llm_text, re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    # Last resort: return stripped text
    return llm_text.strip()


@app.post("/query/sql")
async def natural_language_sql_query(payload: SQLQueryPayload):
    """NL → SQL pipeline: schema-aware context injection → llama3.2 → validation → readonly execution."""
    try:
        schema_context = extract_schema_context()

        system_prompt = (
            "You are a PostgreSQL query generator for an autonomous vehicle spatial telemetry database.\n"
            "You have access to exactly one table. Generate only SELECT queries.\n\n"
            f"{schema_context}\n\n"
            "AV domain rules — read carefully before generating any query:\n\n"
            "1. ego_velocity_vector is a PostgreSQL real[] column (NOT jsonb). It stores [Vx, Vy, Vz] in m/s.\n"
            "   - To filter by ego speed, compute the magnitude using array indexing (1-based in PostgreSQL):\n"
            "     WHERE sqrt(ego_velocity_vector[1]^2 + ego_velocity_vector[2]^2 + ego_velocity_vector[3]^2) > 10.0\n"
            "   - NEVER use raw_telemetry_json to access ego speed. NEVER use ->> or -> on ego_velocity_vector.\n\n"
            "2. detected_objects are inside raw_telemetry_json (jsonb). Use these patterns:\n"
            "   - Filter by classification:  raw_telemetry_json -> 'detected_objects' @> '[{\"classification\": \"car\"}]'\n"
            "   - Filter by occlusion:       raw_telemetry_json -> 'detected_objects' @> '[{\"occlusion_state\": \"total\"}]'\n"
            "   - Count objects per frame:   jsonb_array_length(raw_telemetry_json -> 'detected_objects')\n"
            "   - NEVER use raw_telemetry_json ->> 'classification' — classification is nested inside detected_objects, not a top-level key.\n\n"
            "3. spatial_geometry_embedding is a pgvector column — NEVER include it in SELECT or WHERE.\n\n"
            "Output rules:\n"
            "- Return ONLY a SQL code block inside ```sql ... ``` fences. No explanation, no commentary.\n"
            "- Always end the query with a semicolon.\n"
            "- Use only standard PostgreSQL 14+ syntax.\n"
            "- Limit results to 50 rows maximum unless the user asks for aggregates."
        )

        chat_res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.prompt}
            ]
        )

        raw_llm_output = chat_res.message.content
        if not raw_llm_output:
            raise HTTPException(status_code=502, detail="LLM returned an empty response. Try rephrasing the query.")

        generated_sql = extract_sql_block(raw_llm_output)

        is_valid, rejection_reason = validate_generated_sql(generated_sql)
        if not is_valid:
            return {
                "generated_sql": generated_sql,
                "rows": [],
                "row_count": 0,
                "status": "VALIDATION_FAILED",
                "detail": rejection_reason
            }

        # Execute on a read-only connection — PostgreSQL-level safeguard
        conn = get_conn(register_vec=True)
        try:
            # set_session() requires no open transaction; register_vector() may
            # have opened one, so roll back to a clean state first.
            conn.rollback()
            conn.set_session(readonly=True)
            cursor = conn.cursor()

            try:
                cursor.execute(generated_sql)
                raw_rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
            except Exception as exec_err:
                cursor.close()
                return {
                    "generated_sql": generated_sql,
                    "rows": [],
                    "row_count": 0,
                    "status": "EXECUTION_ERROR",
                    "detail": str(exec_err)
                }

            cursor.close()
        finally:
            put_conn(conn)

        # Embedding column names to suppress (catches aliases via pgvector type check)
        VECTOR_COLS = {"spatial_geometry_embedding"}

        # Serialize rows — convert any non-JSON-serializable types to strings
        serialized_rows = []
        for row in raw_rows:
            serialized_row = {}
            for col, val in zip(col_names, row):
                if col in VECTOR_COLS:
                    serialized_row[col] = "[vector omitted]"
                elif hasattr(val, '__class__') and type(val).__name__ == 'Vector':
                    # Catch pgvector objects that slipped through via an alias
                    serialized_row[col] = "[vector omitted]"
                elif hasattr(val, 'tolist'):
                    serialized_row[col] = val.tolist()
                elif hasattr(val, 'isoformat'):
                    # datetime, date, time objects
                    serialized_row[col] = val.isoformat()
                else:
                    serialized_row[col] = val
            serialized_rows.append(serialized_row)

        return {
            "generated_sql": generated_sql,
            "rows": serialized_rows,
            "row_count": len(serialized_rows),
            "status": "SUCCESS"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/scenario")
async def query_scenario_pipeline(payload: QueryPayload):
    try:
        query_vector = get_spatial_embedding(payload.prompt)

        conn = get_conn(register_vec=True)
        try:
            cursor = conn.cursor()

            # Vector similarity search limited to spatial distance tolerances
            cursor.execute(
                """
                SELECT scene_id, frame_timestamp, raw_telemetry_json, (spatial_geometry_embedding <=> %s::vector) as distance
                FROM spatial_scene_store
                WHERE (spatial_geometry_embedding <=> %s::vector) < 0.75
                ORDER BY distance ASC
                LIMIT 2;
                """,
                (query_vector, query_vector)
            )
            records = cursor.fetchall()
            cursor.close()
        finally:
            put_conn(conn)

        if not records:
            return {
                "answer": r"$$x = [METRIC\_EXTRACTION\_FAILED]$$" + "\n- Status: Fallback Triggered. No spatially relevant context matches found in the vector database layers.",
                "state_vector_seed": None,
                "status": "METRIC_EXTRACTION_FAILED"
            }

        context_blocks = []
        for row in records:
            scene_id = row[0]
            timestamp = row[1]
            raw_json = row[2]

            objects_summary = ""
            for obj in raw_json.get("detected_objects", []):
                objects_summary += (
                    f"- TARGET ID {obj['target_id']} Class: [{obj['classification'].upper()}]\n"
                    f"  Relative Position [X,Y,Z]: {obj['center_xyz']}\n"
                    f"  Size [L,W,H]: {obj['extent_lwh']}\n"
                    f"  Velocity [Vx,Vy,Vz]: {obj['velocity_vector']}\n"
                    f"  Visibility State: {obj['occlusion_state']}\n"
                )

            block = (
                f"DATA SOURCE NODE: {scene_id} at {timestamp:.2f}s\n"
                f"OBJECT LOGS:\n{objects_summary}"
            )
            context_blocks.append(block)

        context_payload = "\n---\n".join(context_blocks)

        system_instructions = (
            "You are a deterministic Autonomous Vehicle Data Extraction Component.\n"
            "CRITICAL COMMANDS:\n"
            "1. Read the provided OBJECT LOGS carefully.\n"
            "2. Output exactly ONE template block tracking the single closest object matching the prompt.\n"
            "3. Do NOT cross-contaminate properties between different target IDs.\n"
            "4. Keep numeric values completely intact. Never alter, extrapolate, or hallucinate dimensions.\n"
            "5. Do NOT include conversational notes or explanations. Start immediately with the template.\n"
            "6. You MUST format the output exactly like the following markdown template block, filling in the properties:\n\n"
            "$$x = [X, Y, V_x, V_y]^T$$\n"
            "- Target ID: [target_id]\n"
            "- Classification: [classification]\n"
            "- Relative Position: X: [X], Y: [Y], Z: [Z]\n"
            "- Velocity Vectors: Vx: [Vx], Vy: [Vy]\n"
            "- Visibility State: [occlusion_state]\n\n"
            f"=== RETRIEVED TELEMETRY CONTEXT BANDS ===\n{context_payload}\n========================================="
        )

        chat_res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": payload.prompt}
            ]
        )

        raw_llm_output = chat_res['message']['content']
        extracted_seed = extract_state_vector(raw_llm_output)

        return {
            "answer": raw_llm_output,
            "state_vector_seed": extracted_seed,
            "status": "SUCCESS" if extracted_seed else "METRIC_EXTRACTION_FAILED"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
