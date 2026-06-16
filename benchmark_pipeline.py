import requests
import json
import time
import numpy as np

# Spatial Telemetry Engine Target Endpoints
INGEST_URL = "http://127.0.0.1:8000/ingest/spatial"
QUERY_URL = "http://127.0.0.1:8000/query/scenario"
timeout_sec = 320
def generate_waymo_sequence(num_frames=100):
    """
    Generates a deterministic time-series sequence mimicking 100 consecutive 
    frames extracted from a Waymo SF Mission tracking log.
    """
    sequence = []
    base_timestamp = 154462720.00
    
    # Simulate a target sedan moving along a continuous tracking vector
    # Real Waymo kinematics: constant velocity with minor noise deviations
    for frame_idx in range(num_frames):
        timestamp = base_timestamp + (frame_idx * 0.1) # 10Hz sampling rate
        ego_v = [8.4 + np.random.normal(0, 0.05), -0.3, 0.0]
        
        # Target car gradually pulling ahead and slightly left
        target_x = 24.8 + (frame_idx * 0.07)
        target_y = -3.1 + (frame_idx * 0.02)
        target_vx = 9.1 + np.random.normal(0, 0.02)
        
        frame_data = {
            "scene_id": "waymo-sf-mission-seq-4192",
            "frame_timestamp": timestamp,
            "ego_velocity_vector": ego_v,
            "camera_extrinsics_rt": [
                [0.999, -0.012, 0.005, 1.43],
                [0.012, 0.998, -0.031, 0.22],
                [-0.005, 0.031, 0.999, 0.85]
            ],
            "detected_objects": [
                {
                    "target_id": 8804,
                    "classification": "sedan",
                    "center_xyz": [target_x, target_y, -0.15],
                    "extent_lwh": [4.8, 1.9, 1.45],
                    "velocity_vector": [target_vx, -0.2, 0.0],
                    "occlusion_state": "partial_500ms" if frame_idx % 10 == 0 else "none"
                }
            ]
        }
        sequence.append(frame_data)
    return sequence

def run_stress_test():
    print("====== INITIALIZING ENGINE METRIC EXTRACTION MATRIX ======")
    frames = generate_waymo_sequence(150)
    
    ingestion_latencies = []
    status_distribution = {}
    
    print(f"[→] Piping {len(frames)} real-world telemetry arrays into backend queue...")
    
    for idx, frame in enumerate(frames):
        try:
            t0 = time.perf_counter()
            response = requests.post(INGEST_URL, json=frame, timeout=timeout_sec)
            t1 = time.perf_counter()
            
            # Track response status codes
            status_code = response.status_code
            status_distribution[status_code] = status_distribution.get(status_code, 0) + 1
            
            if status_code == 200 or status_code == 202:
                ingestion_latencies.append((t1 - t0) * 1000)
            elif idx == 0:
                # Print the first bad response payload to check routing/validation errors
                print(f"[-] Initial Request Failed with Code {status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            status_distribution["CONNECTION_REFUSED"] = status_distribution.get("CONNECTION_REFUSED", 0) + 1
            if idx == 0:
                print(f"[-] Critical: Connection refused at {INGEST_URL}. Is your Uvicorn server actively running?")
                print(f"    Run this first: uvicorn main:app --reload --port 8000")
                
        except Exception as e:
            status_distribution[f"EXCEPTION_{type(e).__name__}"] = status_distribution.get(f"EXCEPTION_{type(e).__name__}", 0) + 1

    # --- Defensive Metric Evaluation Guardrail ---
    print(f"\n[📊 SYSTEM NETWORK DISTRIBUTION]")
    for status, count in status_distribution.items():
        print(f"  • Status [{status}]: {count} frames")

    if not ingestion_latencies:
        print("\n❌ [FATAL] Extraction Aborted: Zero frames were successfully processed by the engine.")
        print("🔍 [ACTION] Verify your engine server state and endpoint routing macros above.")
        return

    # Normal NumPy extraction executes safely only if arrays contain values
    avg_ingest_ms = np.mean(ingestion_latencies)
    max_ingest_ms = np.max(ingestion_latencies)
    p99_ingest_ms = np.percentile(ingestion_latencies, 99)
    
    print(f"\n[📊 INGESTION METRICS LOGGED]")
    print(f"  • Average Network Latency: {avg_ingest_ms:.2f} ms")
    print(f"  • P99 Execution Bound: {p99_ingest_ms:.2f} ms")
    print(f"  • Peak Cluster Overhead: {max_ingest_ms:.2f} ms")
    
    # 2. Evaluate Query Resolution
    print(f"\n[→] Testing Spatial Vector Query Resolution & Parser Accuracy...")
    query_payload = {"prompt": "Find a vehicle sequence matching a sedan roughly 30 meters out with partial occlusion boundaries."}
    
    try:
        t0_query = time.perf_counter()
        q_response = requests.post(QUERY_URL, json=query_payload, timeout=timeout_sec)
        t1_query = time.perf_counter()
        
        query_ms = (t1_query - t0_query) * 1000
        res_json = q_response.json()
        
        print(f"\n[📊 RETRIEVED QUERY METRICS LOGGED]")
        print(f"  • Total Subsystem Latency: {query_ms:.2f} ms")
        print(f"  • Raw Model Text Output:\n{res_json.get('answer')}")
        print(f"  • Dynamic Extraction Status: {res_json.get('status')}")
        print(f"  • Hydrated Seed Matrix: {res_json.get('state_vector_seed')}")
    except Exception as e:
        print(f"[-] Query endpoint tracking failed: {e}")

if __name__ == "__main__":
    run_stress_test()
