import requests
import json
import time

# Target endpoint to our async engine core
INGEST_URL = "http://127.0.0.1:8000/ingest/spatial"

def dispatch_mock_waymo_frame():
    # Waymo specific tracking run parameters 
    waymo_payload = {
        "scene_id": "waymo-sf-mission-seq-4192",
        "frame_timestamp": 154462718.410,
        "ego_velocity_vector": [8.4, -0.3, 0.0],
        "camera_extrinsics_rt": [
            [0.999, -0.012, 0.005, 1.43],
            [0.012, 0.998, -0.031, 0.22],
            [-0.005, 0.031, 0.999, 0.85]
        ],
        "detected_objects": [
            {
                "target_id": 8804,
                "classification": "sedan",
                "center_xyz": [24.8, -3.1, -0.15], # Waymo relative distance coordinates
                "extent_lwh": [4.8, 1.9, 1.45],
                "velocity_vector": [9.1, -0.2, 0.0],
                "occlusion_state": "partial_500ms"
            },
            {
                "target_id": 9102,
                "classification": "pedestrian",
                "center_xyz": [8.2, 4.5, -0.6],
                "extent_lwh": [0.6, 0.7, 1.7],
                "velocity_vector": [0.2, 1.1, 0.0],
                "occlusion_state": "none"
            }
        ]
    }

    print(f"[→] Dispatching Waymo Sequence Frame to Local Ingestion Loop...")
    start_time = time.time()
    
    response = requests.post(INGEST_URL, json=waymo_payload)
    latency = (time.time() - start_time) * 1000

    print(f"[←] Server Response Received in {latency:.2f}ms")
    print(f"Status Code: {response.status_code}")
    print(f"Payload Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    dispatch_mock_waymo_frame()
