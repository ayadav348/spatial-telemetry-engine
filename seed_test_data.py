"""
seed_test_data.py
=================
Generates and ingests ~200 synthetic-but-realistic AV telemetry frames covering
edge cases that the existing inline test data does NOT exercise. The goal is to
give the NL->SQL pipeline a rich, diverse dataset so query coverage and edge
cases can be tested meaningfully.

Each frame conforms exactly to the SpatialFramePayload schema enforced by
main.py:
    scene_id            : str
    frame_timestamp     : float
    ego_velocity_vector : [Vx, Vy, Vz]      (3 floats, m/s)
    camera_extrinsics_rt: 3x4 row-major matrix
    detected_objects    : list of BoundingBox3D:
        target_id, classification, center_xyz[3], extent_lwh[3],
        velocity_vector[3], occlusion_state

Usage:
    1. Start the server:  uvicorn main:app --port 8000
    2. Run:               python seed_test_data.py

The script POSTs every frame to /ingest/spatial, waits for the background
workers to drain, then queries /db/stats to verify the final frame count.
"""

import time
import math
import requests

BASE_URL = "http://127.0.0.1:8000"
INGEST_URL = f"{BASE_URL}/ingest/spatial"
STATS_URL = f"{BASE_URL}/db/stats"

# Identity-ish extrinsics reused across frames (the pipeline does not depend on it).
IDENTITY_EXTRINSICS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]


def _obj(target_id, classification, center, extent, vel, occlusion="none"):
    """Build a single BoundingBox3D-compatible detected object dict."""
    return {
        "target_id": target_id,
        "classification": classification,
        "center_xyz": [round(float(c), 3) for c in center],
        "extent_lwh": [round(float(e), 3) for e in extent],
        "velocity_vector": [round(float(v), 3) for v in vel],
        "occlusion_state": occlusion,
    }


def _frame(scene_id, ts, ego_v, objects):
    """Build a single SpatialFramePayload-compatible frame dict."""
    return {
        "scene_id": scene_id,
        "frame_timestamp": round(float(ts), 3),
        "ego_velocity_vector": [round(float(v), 3) for v in ego_v],
        "camera_extrinsics_rt": IDENTITY_EXTRINSICS,
        "detected_objects": objects,
    }


# ---------------------------------------------------------------------------
# Edge-case scenario generators. Each returns a list of frames.
# ---------------------------------------------------------------------------

def gen_zero_objects(n=10):
    """Frames with NO detected objects -> tests jsonb_array_length(...) = 0."""
    frames = []
    for i in range(n):
        frames.append(_frame(
            "edge-zero-objects",
            1000.0 + i * 0.1,
            [12.0, 0.0, 0.0],
            [],  # empty detection list
        ))
    return frames


def gen_multi_object_types(n=30):
    """Each frame contains car + pedestrian + cyclist + truck.
    Tests classification filtering and per-frame object counts."""
    frames = []
    for i in range(n):
        ts = 2000.0 + i * 0.1
        objs = [
            _obj(100 + i, "car",        [15.0 + i * 0.1, -2.0, -0.1], [4.6, 1.9, 1.5], [9.0, 0.1, 0.0]),
            _obj(200 + i, "pedestrian", [6.0, 3.5, -0.7],            [0.6, 0.6, 1.8], [0.5, 1.2, 0.0]),
            _obj(300 + i, "cyclist",    [10.0, -4.0, -0.4],          [1.8, 0.6, 1.6], [4.0, 0.0, 0.0]),
            _obj(400 + i, "truck",      [40.0, 1.0, 0.2],            [12.0, 2.6, 3.5], [8.5, 0.0, 0.0], "partial_500ms"),
        ]
        frames.append(_frame("edge-multi-object-types", ts, [10.5, -0.2, 0.0], objs))
    return frames


def gen_full_occlusion(n=20):
    """All objects fully occluded -> tests occlusion_state = 'total' queries."""
    frames = []
    for i in range(n):
        ts = 3000.0 + i * 0.1
        objs = [
            _obj(500 + i, "car",   [20.0, -1.0, -0.1], [4.7, 1.9, 1.5], [7.0, 0.0, 0.0], "total"),
            _obj(600 + i, "truck", [55.0, 2.0, 0.3],   [11.0, 2.5, 3.4], [6.0, 0.0, 0.0], "total"),
        ]
        frames.append(_frame("edge-full-occlusion", ts, [9.0, 0.0, 0.0], objs))
    return frames


def gen_highway_highspeed(n=40):
    """Ego on highway ~30 m/s -> tests ego-speed magnitude filtering."""
    frames = []
    for i in range(n):
        ts = 4000.0 + i * 0.1
        ego = [29.5 + (i % 5) * 0.2, -0.5, 0.0]  # ~30 m/s
        objs = [
            _obj(700 + i, "car", [50.0 - i * 0.2, -3.5, -0.1], [4.8, 1.9, 1.5], [30.0, 0.0, 0.0]),
        ]
        frames.append(_frame("edge-highway-highspeed", ts, ego, objs))
    return frames


def gen_parked_ego(n=15):
    """Ego stationary ~0 m/s -> tests speed ≈ 0 edge case."""
    frames = []
    for i in range(n):
        ts = 5000.0 + i * 0.1
        objs = [
            _obj(800 + i, "pedestrian", [3.0, 1.0, -0.7], [0.6, 0.6, 1.8], [0.8, 0.4, 0.0]),
        ]
        frames.append(_frame("edge-parked-ego", ts, [0.0, 0.0, 0.0], objs))
    return frames


def gen_extreme_range(n=25):
    """Objects at extreme close (1m) and far (80m) ranges -> distance filtering."""
    frames = []
    for i in range(n):
        ts = 6000.0 + i * 0.1
        objs = [
            _obj(900 + i, "pedestrian", [1.0, 0.2, -0.7],  [0.6, 0.6, 1.8], [0.0, 0.0, 0.0]),
            _obj(950 + i, "truck",      [80.0, -1.0, 0.3], [12.0, 2.6, 3.6], [12.0, 0.0, 0.0]),
        ]
        frames.append(_frame("edge-extreme-range", ts, [14.0, 0.0, 0.0], objs))
    return frames


def gen_approaching_objects(n=30):
    """Objects with negative Vx (closing in on ego) -> velocity-direction queries."""
    frames = []
    for i in range(n):
        ts = 7000.0 + i * 0.1
        # Object starts far and approaches; negative velocity along X.
        x = 60.0 - i * 1.5
        objs = [
            _obj(1000 + i, "car", [max(x, 5.0), 0.5, -0.1], [4.7, 1.9, 1.5], [-15.0, 0.0, 0.0],
                 "partial_500ms" if x < 25 else "none"),
        ]
        frames.append(_frame("edge-approaching-objects", ts, [8.0, 0.0, 0.0], objs))
    return frames


def gen_long_sequence(n=30):
    """Single scene, many frames -> tests aggregation / GROUP BY across rows."""
    frames = []
    for i in range(n):
        ts = 8000.0 + i * 0.1
        objs = [
            _obj(1100, "car", [25.0 + i * 0.05, -3.0 + i * 0.01, -0.15],
                 [4.8, 1.9, 1.45], [9.1, -0.2, 0.0],
                 "partial_500ms" if i % 10 == 0 else "none"),
        ]
        frames.append(_frame("edge-long-sequence", ts, [8.4, -0.3, 0.0], objs))
    return frames


def build_all_frames():
    generators = [
        gen_zero_objects,        # 10
        gen_multi_object_types,  # 30
        gen_full_occlusion,      # 20
        gen_highway_highspeed,   # 40
        gen_parked_ego,          # 15
        gen_extreme_range,       # 25
        gen_approaching_objects, # 30
        gen_long_sequence,       # 30
    ]
    frames = []
    for gen in generators:
        frames.extend(gen())
    return frames


def get_frame_count():
    try:
        r = requests.get(STATS_URL, timeout=10)
        return r.json().get("frame_count", -1)
    except Exception as e:
        print(f"[-] Could not read /db/stats: {e}")
        return -1


def main():
    print("====== SPATIAL TELEMETRY EDGE-CASE SEEDER ======")

    # Pre-flight: confirm the server is reachable.
    start_count = get_frame_count()
    if start_count < 0:
        print(f"[FATAL] Cannot reach the engine at {BASE_URL}.")
        print("        Start it first:  uvicorn main:app --port 8000")
        return
    print(f"[i] Starting frame count in DB: {start_count}")

    frames = build_all_frames()
    print(f"[i] Generated {len(frames)} edge-case frames across 8 scenarios.")

    accepted = 0
    failed = 0
    for idx, frame in enumerate(frames):
        try:
            resp = requests.post(INGEST_URL, json=frame, timeout=30)
            if resp.status_code in (200, 202):
                accepted += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"[-] Frame {idx} rejected ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"[-] Frame {idx} exception: {e}")

    print(f"[i] HTTP accepted: {accepted}, failed: {failed}")

    # Ingestion is offloaded to background workers; poll until the count
    # stabilizes at the expected target (or we time out).
    expected = start_count + accepted
    print(f"[i] Waiting for background workers to drain (target {expected} frames)...")
    deadline = time.time() + 60
    final_count = start_count
    while time.time() < deadline:
        final_count = get_frame_count()
        if final_count >= expected:
            break
        time.sleep(1.0)

    print("\n[ VERIFICATION ]")
    print(f"  • Frames before seeding : {start_count}")
    print(f"  • HTTP-accepted frames  : {accepted}")
    print(f"  • Expected final count  : {expected}")
    print(f"  • Actual final count    : {final_count}")
    if final_count >= expected:
        print("  ✅ SUCCESS — all accepted frames are indexed in the DB.")
    else:
        print(f"  ⚠️  MISMATCH — {expected - final_count} frames not yet visible "
              "(workers may still be processing, or some inserts failed).")


if __name__ == "__main__":
    main()
