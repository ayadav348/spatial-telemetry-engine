import os
import subprocess
import time
import socket
import json
import math
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg2
from pgvector.psycopg2 import register_vector
# import ollama
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Spatial Telemetry Engine", description="Local 3D Scene Discovery & Volumetric Retrieval Platform")

# Configured to target your newly initialized PostgreSQL layer
DB_PARAMS = {
    "dbname": "spatial_vector_db",
    "user": "postgres",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

# LLM_MODEL = "llama3.2"
# EMBED_MODEL = "nomic-embed-text"

# --- High-Velocity Native Local Embedding Layer ---
SPATIAL_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def get_spatial_embedding(text_payload: str):
    """Generates dense vector matrices instantly on local hardware."""
    vector_array = SPATIAL_EMBEDDER.encode(text_payload)
    return vector_array.tolist()

# def bootstrap_ollama():
#     """Checks if the local Ollama port is open. If closed, automates engine boot."""
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         is_running = s.connect_ex(('127.0.0.1', 11434)) == 0
# 
#     if not is_running:
#         print("[*] Ollama daemon is offline. Initializing automatic background boot thread...")
#         try:
#             subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             time.sleep(3)  # Allow time slice for socket allocation
#         except Exception as e:
#             print(f"[-] Auto-boot failed. Attempting alternative systemd invoke: {e}")
#             subprocess.Popen(["sudo", "systemctl", "start", "ollama"])
#             time.sleep(3)
# 
#     try:
#         local_models = [m['model'] for m in ollama.list().get('models', [])]
#         if f"{EMBED_MODEL}:latest" not in local_models and EMBED_MODEL not in local_models:
#             print(f"[*] Fetching required embedding matrix: {EMBED_MODEL}")
#             ollama.pull(EMBED_MODEL)
#         if f"{LLM_MODEL}:latest" not in local_models and LLM_MODEL not in local_models:
#             print(f"[*] Fetching required synthesis matrix: {LLM_MODEL}")
#             ollama.pull(LLM_MODEL)
#         print("[+] Ollama execution layer online and verified.")
#     except Exception as e:
#         print(f"[-] Target verification error: {e}. Confirm manually via 'ollama list'")
# 
# # Fire daemon automation layer
# bootstrap_ollama()

def init_db():
    """Initializes extension registers and spatial telemetry store tables."""
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    # Mutate table schema to explicitly hold structural physical states
    # Note: Vector dimensions dropped to 384 to mirror all-MiniLM-L6-v2 footprints
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
    cursor.close()
    conn.close()

init_db()

# --- Pydantic Validation Models for Structural Telemetry Ingestion ---

class BoundingBox3D(BaseModel):
    target_id: int
    classification: str
    center_xyz: List[float] = Field(..., min_length=3, max_length=3)  # [X, Y, Z] relative coordinates
    extent_lwh: List[float] = Field(..., min_length=3, max_length=3)  # [Length, Width, Height]
    velocity_vector: List[float] = Field(..., min_length=3, max_length=3) # [Vx, Vy, Vz]
    occlusion_state: str # e.g., "none", "partial_500ms", "total"

class SpatialFramePayload(BaseModel):
    scene_id: str
    frame_timestamp: float
    ego_velocity_vector: List[float] = Field(..., min_length=3, max_length=3) # [Vx, Vy, Vz]
    camera_extrinsics_rt: List[List[float]] # 3x4 or 4x4 matrix transformation rows
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

# --- Spatial Math Serialization Utility ---

def serialize_spatial_frame(payload: SpatialFramePayload) -> str:
    """Converts multi-dimensional matrix inputs and coordinate tracking telemetry into a string representation."""
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

# --- Asynchronous Worker Logic ---

def process_and_store_telemetry(payload: SpatialFramePayload):
    """Background task function handling serialization, native embedding, and DB insertion."""
    try:
        serialized_context = serialize_spatial_frame(payload)

        # Generate spatial feature embedding coordinates via native sentence-transformers
        spatial_vector = get_spatial_embedding(serialized_context)

        # --- Old Ollama Ingestion Code Block ---
        # response = ollama.embed(model=EMBED_MODEL, input=serialized_context)
        # if 'embeddings' in response and response['embeddings']:
        #     spatial_vector = response['embeddings'][0]
        # elif 'embedding' in response:
        #     spatial_vector = response['embedding']
        # else:
        #     print("[-] Critical Error: Ollama output missing valid vector matrix types.")
        #     return

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)

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
                json.dumps(payload.dict()),
                spatial_vector
            )
        )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[+] Asynchronously indexed tracking metrics for sequence frame: {payload.scene_id}")
    except Exception as e:
        print(f"[-] Background Ingestion Thread Exception: {str(e)}")

# --- REST Endpoints ---

@app.post("/ingest/spatial", status_code=status.HTTP_202_ACCEPTED)
async def ingest_spatial_telemetry(payload: SpatialFramePayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_and_store_telemetry, payload)
    return {
        "status": "QUEUED",
        "scene_id": payload.scene_id,
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
    """
    Parses strict LaTeX state-space arrays from local LLM context blocks.
    Falls back to explicit text parameter scanning if the math block contains literal placeholders.
    """
    # Pass 1: Standard LaTeX check for active digit arrays: x = [1.2, 3.4, ...]^T
    math_pattern = r"x\s*=\s*\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)(?:,\s*(-?\d+(?:\.\d+)?))?\s*\]\^T"
    match = re.search(math_pattern, llm_text)
    if match:
        extracted = [float(c) for c in match.groups() if c is not None]
        if extracted:
            return extracted

    # Pass 2: Fallback Parameter Extraction (Scans the rigid bullet points populated by the LLM)
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

@app.post("/query/scenario")
async def query_scenario_pipeline(payload: QueryPayload):
    try:
        # Generate dense prompt vector directly on local hardware via Sentence Transformers
        query_vector = get_spatial_embedding(payload.prompt)

        # --- Old Ollama Query Code Block ---
        # response = ollama.embed(model=EMBED_MODEL, input=payload.prompt)
        # if 'embeddings' in response and response['embeddings']:
        #     query_vector = response['embeddings'][0]
        # else:
        #     query_vector = response['embedding']

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)

        cursor.execute(
            """
            SELECT scene_id, frame_timestamp, raw_telemetry_json
            FROM spatial_scene_store
            ORDER BY spatial_geometry_embedding <=> %s::vector
            LIMIT 3;
            """,
            (query_vector,)
        )
        records = cursor.fetchall()
        cursor.close()
        conn.close()

        if not records:
            return {
                "answer": "No indexed scene matrices discovered in spatial database layers.",
                "state_vector_seed": None,
                "status": "EMPTY_DATABASE"
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
                    f"  Position [X,Y,Z]: {obj['center_xyz']}\n"
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
            "1. Output ONLY exact parameters directly printed in the OBJECT LOGS context.\n"
            "2. Keep numeric values completely intact. Never alter them.\n"
            "3. Do NOT include conversational notes, greetings, explanations, or introductory text.\n"
            "4. You MUST format the output exactly like the following markdown template block, filling in the bracketed properties.\n"
            "5. The state vector array inside the brackets MUST contain EXACTLY four numeric values: [X, Y, Vx, Vy]. If Vy is missing, pad it with 0.0:\n\n"
            "$$x = [X, Y, V_x, V_y]^T$$\n"
            "- Target ID: [target_id]\n"
            "- Classification: [classification]\n"
            "- Relative Position: X: [X], Y: [Y], Z: [Z]\n"
            "- Velocity Vectors: Vx: [Vx], Vy: [Vy]\n"
            "- Visibility State: [occlusion_state]\n\n"
            f"=== RETRIEVED TELEMETRY CONTEXT BANDS ===\n{context_payload}\n========================================="
        )        
        
        # NOTE: Leaving this here as it requires an active, responding local llama-server instance.
        # If your local Ollama daemon hangs or times out on ROCm, substitute this with a remote Groq endpoint.
        import ollama
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
