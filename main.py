import os
import subprocess
import time
import socket
import json
import math
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama

app = FastAPI(title="Spatial Telemetry Engine", description="Local 3D Scene Discovery & Volumetric Retrieval Platform")

# Configured to target your newly initialized PostgreSQL layer
DB_PARAMS = {
    "dbname": "spatial_vector_db",
    "user": "postgres",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

def bootstrap_ollama():
    """Checks if the local Ollama port is open. If closed, automates engine boot."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex(('127.0.0.1', 11434)) == 0

    if not is_running:
        print("[*] Ollama daemon is offline. Initializing automatic background boot thread...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Allow time slice for socket allocation
        except Exception as e:
            print(f"[-] Auto-boot failed. Attempting alternative systemd invoke: {e}")
            subprocess.Popen(["sudo", "systemctl", "start", "ollama"])
            time.sleep(3)

    try:
        local_models = [m['model'] for m in ollama.list().get('models', [])]
        if f"{EMBED_MODEL}:latest" not in local_models and EMBED_MODEL not in local_models:
            print(f"[*] Fetching required embedding matrix: {EMBED_MODEL}")
            ollama.pull(EMBED_MODEL)
        if f"{LLM_MODEL}:latest" not in local_models and LLM_MODEL not in local_models:
            print(f"[*] Fetching required synthesis matrix: {LLM_MODEL}")
            ollama.pull(LLM_MODEL)
        print("[+] Ollama execution layer online and verified.")
    except Exception as e:
        print(f"[-] Target verification error: {e}. Confirm manually via 'ollama list'")

# Fire daemon automation layer
bootstrap_ollama()

def init_db():
    """Initializes extension registers and spatial telemetry store tables."""
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    # Mutate table schema to explicitly hold structural physical states
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spatial_scene_store (
            id serial PRIMARY KEY,
            scene_id text NOT NULL,
            frame_timestamp real NOT NULL,
            ego_velocity_vector real[] NOT NULL,
            raw_telemetry_json jsonb NOT NULL,
            spatial_geometry_embedding vector(768)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# --- Pydantic Validation Models for Structural Telemetry Ingestion (V2 Standards Complete) ---

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

# Added validation model to support incoming parameters from the UI dataset bridge
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
    """
    Converts multi-dimensional matrix inputs and coordinate tracking telemetry
    into a dense structural string representation for the embedding layer.
    """
    ego_speed = (sum(v**2 for v in payload.ego_velocity_vector))**0.5

    # Base structural metadata anchor
    structural_string = (
        f"Sequence frame token: {payload.scene_id}. Physical Timestamp offsets: {payload.frame_timestamp:.3f}s. "
        f"Ego vehicle linear velocity state magnitude: {ego_speed:.2f} m/s. "
    )

    # Serialize target bounding box trajectories relative to ego origin
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

# --- REST Endpoints ---

@app.post("/ingest/spatial")
async def ingest_spatial_telemetry(payload: SpatialFramePayload):
    try:
        # Convert raw numerical matrices into a dense semantic topology map
        serialized_context = serialize_spatial_frame(payload)

        # Generate spatial feature embedding coordinates
        response = ollama.embed(model=EMBED_MODEL, input=serialized_context)
        spatial_vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)

        # Store explicit spatial variables alongside raw structural jsonb tracking logs
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
            )
        )

        conn.commit()
        cursor.close()
        conn.close()

        print(f"[+] Successfully indexed tracking metrics for sequence frame: {payload.scene_id}")
        return {"status": "success", "scene_indexed": payload.scene_id, "frame_timestamp": payload.frame_timestamp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New Endpoint implementation to fix the 404 error
@app.post("/ingest/dataset-bridge")
async def ingest_through_dataset_bridge(payload: DatasetBridgePayload):
    try:
        # 1. Coordinate Normalization (Absolute World Pos -> Local Relative to Ego)
        relative_center = [
            payload.box_translation[0] - payload.ego_translation[0],
            payload.box_translation[1] - payload.ego_translation[1],
            payload.box_translation[2] - payload.ego_translation[2]
        ]

        # 2. Re-wrap into native internal layout schema
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

        # 3. Direct pass-through execution into vector serialization loop
        return await ingest_spatial_telemetry(native_payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset Transformation Exception: {str(e)}")

@app.post("/query/scenario")
async def query_scenario_pipeline(payload: QueryPayload):
    try:
        # Embed the spatial tracking constraint request
        response = ollama.embed(model=EMBED_MODEL, input=payload.prompt)
        query_vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)

        # Query the database for the 3 closest spatial layouts using cosine distance
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
            return {"answer": "No indexed scene matrices discovered in spatial database layers."}

        # Build context payloads by isolating individual target properties into an explicit schema map
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

        # Rigidly lock down the small LLM's role to prevent semantic merging and field re-writing
 # Rigidly lock down the small LLM's role to prevent semantic merging and field re-writing
        system_instructions = (
            "You are a deterministic Autonomous Vehicle Data Extraction Component.\n"
            "CRITICAL COMMANDS:\n"
            "1. Output ONLY exact parameters directly printed in the OBJECT LOGS context.\n"
            "2. Keep numeric values completely intact. Never alter them.\n"
            "3. Do NOT include conversational notes, greetings, explanations, or introductory text.\n"
            "4. You MUST format the output exactly like the following markdown template block, filling in the bracketed properties:\n\n"
            "$$x = [X, Y, V_x, V_y]^T$$\n"
            "- Target ID: [target_id]\n"
            "- Classification: [classification]\n"
            "- Relative Position: X: [X], Y: [Y], Z: [Z]\n"
            "- Velocity Vectors: Vx: [Vx], Vy: [Vy]\n"
            "- Visibility State: [occlusion_state]\n\n"
            f"=== RETRIEVED TELEMETRY CONTEXT BANDS ===\n{context_payload}\n========================================="
        )
        chat_res = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": payload.prompt}
            ]
        )

        return {"answer": chat_res['message']['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
