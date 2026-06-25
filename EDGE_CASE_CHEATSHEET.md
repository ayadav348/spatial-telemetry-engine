# Edge Case Cheat Sheet — Spatial Telemetry Engine

## How to Read This Document

This cheat sheet maps every scene currently loaded in the database to the exact data values inside it.

**Key concepts:**

- **scene_id** — the logical name for a test scenario. Each scene contains multiple frames.
- **frame** — a single telemetry snapshot. Frames are 0.1s apart (10 Hz).
- **ego** — the vehicle carrying the sensor. Its velocity is `[Vx, Vy, Vz]` in m/s, where Vx is forward.
- **detected_objects** — list of objects the sensor saw in that frame. Can be empty.
- **center_xyz** — object position relative to the ego vehicle in meters `[forward, left/right, up/down]`. Positive X = in front, positive Y = left.
- **velocity_vector** — object's velocity in m/s in the same frame. **Negative Vx means the object is moving toward the ego.**
- **extent_lwh** — object bounding box size `[length, width, height]` in meters.
- **occlusion_state** — how obscured the object is: `"none"`, `"partial_500ms"`, or `"total"`.
- **target_id** — integer identifier for a detected object. Not guaranteed to be stable across frames (depends on the scene).

**How to use the "Test for" rows:** each entry describes a specific behavior or failure mode that the scene was designed to exercise. When you write a test, pick the scene that puts the system in that exact condition.

---

## DB at a Glance

| Property | Value |
|---|---|
| Database | `spatial_vector_db` (PostgreSQL, localhost:5432) |
| Table | `spatial_scene_store` |
| Total rows | 203 |
| Total scenes | 9 |
| Frame cadence | 0.1s steps |
| Embedding dim | 384 (all-MiniLM-L6-v2) |
| Camera extrinsics | Identity matrix in all scenes (no rotation/translation) |

---

## Scene Inventory

### 1. `edge-zero-objects`
**10 frames | t: 1000.0 – 1000.9s**

| Field | Value |
|---|---|
| Ego velocity | `[12.0, 0.0, 0.0]` m/s — constant |
| Detected objects | **`[]` empty every frame** |
| Object count | 0 |
| **Test for** | Empty detection list handling, null-safe aggregation, vector embedding of empty scenes, queries that count or filter objects when none exist |

---

### 2. `edge-multi-object-types`
**30 frames | t: 2000.0 – 2002.9s**

Ego velocity: `[10.5, -0.2, 0.0]` m/s (constant). 4 objects per frame, one of each class.

| target_id | Class | Position (x, y, z) | Extent (l, w, h) | Velocity | Occlusion |
|---|---|---|---|---|---|
| 100+n | **car** | [15.0+0.1n, -2.0, -0.1] | [4.6, 1.9, 1.5] | [9.0, 0.1, 0.0] | none |
| 200+n | **pedestrian** | [6.0, 3.5, -0.7] | [0.6, 0.6, 1.8] | [0.5, 1.2, 0.0] | none |
| 300+n | **cyclist** | [10.0, -4.0, -0.4] | [1.8, 0.6, 1.6] | [4.0, 0.0, 0.0] | none |
| 400+n | **truck** | [40.0, 1.0, 0.2] | [12.0, 2.6, 3.5] | [8.5, 0.0, 0.0] | **partial_500ms** |

> `n` = frame index (0–29). The car's X position advances 0.1m per frame; all other objects are static in position.

**Test for:** Multi-class filtering, mixed occlusion queries, class-specific aggregation, the only scene with a cyclist, truck at 40m with partial occlusion alongside clear near-field objects.

---

### 3. `edge-full-occlusion`
**20 frames | t: 3000.0 – 3001.9s**

Ego velocity: `[9.0, 0.0, 0.0]` m/s (constant). Every object is `total` occlusion in every frame.

| target_id | Class | Position (x, y, z) | Velocity | Occlusion |
|---|---|---|---|---|
| 500+n | **car** | [20.0, -1.0, -0.1] | [7.0, 0.0, 0.0] | **total** |
| 600+n | **truck** | [55.0, 2.0, 0.3] | [6.0, 0.0, 0.0] | **total** |

**Test for:** Scenes where all detections are fully occluded, occlusion-filtered queries that should return 0 visible objects, behavior when the system has detections but no line-of-sight to any of them.

---

### 4. `edge-highway-highspeed`
**40 frames | t: 4000.0 – 4003.9s**

| Field | Value |
|---|---|
| Ego velocity | `[29.5–30.3, -0.5, 0.0]` m/s — varies slightly across frames |
| Object (700+n) | **car**, x: 42.2–50.0m, Vx: **30.0** m/s (constant), occlusion: none |
| Closing speed | ~0 m/s (both ego and object travelling at ~30 m/s in same direction) |

**Test for:** High absolute speeds, near-zero closing speed despite fast ego, float precision at 30+ m/s, non-zero ego Vy (-0.5 m/s lateral drift), leading car scenario on highway.

---

### 5. `edge-parked-ego`
**15 frames | t: 5000.0 – 5001.4s**

| Field | Value |
|---|---|
| Ego velocity | **`[0.0, 0.0, 0.0]`** — fully stationary, every frame |
| Object (800) | **pedestrian**, pos: `[3.0, 1.0, -0.7]` (static across all frames), Vel: `[0.8, 0.4, 0.0]` |

> The pedestrian position does not change across frames — the pedestrian velocity is recorded but the seeded position is fixed. The pedestrian is moving but the snapshot position is constant (3m ahead, 1m left).

**Test for:** Zero ego speed edge case, relative velocity when ego denominator is 0, TTC calculation with stationary ego, a moving pedestrian at very close range (3m).

---

### 6. `edge-extreme-range`
**25 frames | t: 6000.0 – 6002.4s**

Ego velocity: `[14.0, 0.0, 0.0]` m/s (constant). Two objects at opposite distance extremes simultaneously.

| target_id | Class | Position (x, y, z) | Extent (l, w, h) | Velocity | Occlusion |
|---|---|---|---|---|---|
| 900+n | **pedestrian** | [**1.0**, 0.2, -0.7] | [0.6, 0.6, 1.8] | [0.0, 0.0, 0.0] | none |
| 950+n | **truck** | [**80.0**, -1.0, 0.3] | [12.0, 2.6, 3.6] | [12.0, 0.0, 0.0] | none |

**Test for:** Minimum range (1m — near field collision boundary), maximum range (80m — sensor limit), range-based query filtering at boundaries, near-field bounding box overlap with ego, simultaneous near+far object handling.

---

### 7. `edge-approaching-objects`
**30 frames | t: 7000.0 – 7002.9s**

| Field | Value |
|---|---|
| Ego velocity | `[8.0, 0.0, 0.0]` m/s (constant) |
| Object (1000–1029) | **car**, Vx: **-15.0** m/s (head-on, closing toward ego) |
| Object X position | 60.0m → 16.5m (decreases 1.5m per frame) |
| Closing speed | 8 + 15 = **23 m/s** |
| Estimated TTC at t=7000 | ~60m / 23 m/s ≈ **2.6 seconds** |

> Each frame has a new target_id (1000, 1001, ... 1029) — object identity is not persistent across frames in this scene.

**Test for:** Negative Vx detection and handling, closing speed calculation, TTC computation, head-on collision scenario, rapidly decreasing object range, non-persistent target IDs.

---

### 8. `edge-long-sequence`
**30 frames | t: 8000.0 – 8002.9s**

| Field | Value |
|---|---|
| Ego velocity | `[8.4, -0.3, 0.0]` m/s (constant) |
| Object | **car** ID **1100** — same target_id all 30 frames |
| Object X position | 25.0m → 26.45m (advances 0.05m per frame) |
| Occlusion pattern | `partial_500ms` at t=8000, 8001, 8002 (every 10th frame / every 1.0s); `none` all others |

**Test for:** Long time-series aggregation on a single persistent target, periodic occlusion (occlusion fires exactly every 10 frames), tracking continuity when occlusion interrupts, queries that aggregate across a full 3-second window.

---

### 9. `nuscenes-log-segment-081`
**3 frames | t: 44.102s (all three frames share the same timestamp)**

| Field | Value |
|---|---|
| Ego velocity | `[8.4, 1.2, 0.0]` m/s |
| Object | **truck** ID 4022, pos: `[11.7, 4.6, 0.2]`, Vel: `[-2.1, 0.5, 0.0]`, occlusion: none |
| Source | Ingested via the dataset bridge (nuScenes raw format → normalized coordinates) |
| Quirk | All 3 rows have identical timestamps — this was ingested 3 times from the UI |

**Test for:** Dataset bridge normalization correctness, duplicate timestamp handling, nuScenes-origin data round-trip, closing truck at close range (11.7m, Vx=-2.1 so object moving toward ego).

---

## Quick Reference: Value Extremes

| Metric | Min | Max | Scene |
|---|---|---|---|
| Ego speed (Vx) | 0.0 m/s | 30.3 m/s | `edge-parked-ego` / `edge-highway-highspeed` |
| Object Vx | -15.0 m/s (closing) | +30.0 m/s | `edge-approaching-objects` / `edge-highway-highspeed` |
| Object range (x) | 1.0 m | 80.0 m | `edge-extreme-range` |
| Objects per frame | 0 | 4 | `edge-zero-objects` / `edge-multi-object-types` |
| Frames per scene | 10 | 40 | `edge-zero-objects` / `edge-highway-highspeed` |
| Sequence duration | 1.0s | 3.9s | `edge-zero-objects` / `edge-highway-highspeed` |
| Unique target IDs in scene | 1 (ID 1100) | 30 (IDs 1000–1029) | `edge-long-sequence` / `edge-approaching-objects` |

## Occlusion States in the DB

| State | Meaning | Where |
|---|---|---|
| `"none"` | Fully visible | Most scenes |
| `"partial_500ms"` | Partially occluded, seen within last 500ms | `edge-multi-object-types` (truck), `edge-long-sequence` (periodic) |
| `"total"` | Completely occluded — no line of sight | `edge-full-occlusion` only (all objects, all frames) |

## Object Classes in the DB

| Class | Scenes |
|---|---|
| `car` | multi-object-types, full-occlusion (n/a), highway-highspeed, long-sequence, approaching-objects, extreme-range (n/a) |
| `pedestrian` | multi-object-types, parked-ego, extreme-range |
| `cyclist` | multi-object-types only |
| `truck` | multi-object-types, full-occlusion, extreme-range, nuscenes-log-segment-081 |
