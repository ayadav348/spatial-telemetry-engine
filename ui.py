import streamlit as st
import requests
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(layout="wide", page_title="Spatial Telemetry Dashboard")

API_BASE = "http://127.0.0.1:8000"

# Color per object class for the top-down BEV diagram.
CLASS_COLORS = {
    "car": "#e63946", "sedan": "#e63946", "truck": "#9d0208", "vehicle": "#e63946",
    "pedestrian": "#2a9d8f", "person": "#2a9d8f",
    "cyclist": "#e9a000", "bicycle": "#e9a000", "motorcycle": "#e9a000",
    "sign": "#6c757d", "unknown": "#6c757d",
}


def _class_color(classification: str) -> str:
    return CLASS_COLORS.get((classification or "").lower(), "#6c757d")


def render_bev_frame(frame: dict, view_range: float = 60.0):
    """Draws a top-down bird's-eye-view of one telemetry frame.

    Ego vehicle sits at the origin (0, 0) looking along +X (up). Each detected
    object is placed at its ego-relative (X, Y) with a velocity arrow. Returns a
    matplotlib Figure ready for st.pyplot().
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Range rings every 10 m so distances are readable at a glance.
    for r in range(10, int(view_range) + 1, 10):
        ax.add_artist(plt.Circle((0, 0), r, fill=False, color="#dddddd", lw=0.8, zorder=0))
        ax.text(0, r, f"{r}m", color="#bbbbbb", fontsize=7, ha="center", va="bottom", zorder=0)

    # Ego vehicle: a rectangle at the origin pointing up (+X is forward).
    ego_l, ego_w = 4.5, 2.0
    ax.add_patch(Rectangle((-ego_w / 2, -ego_l / 2), ego_w, ego_l,
                           color="#1d3557", zorder=5))
    ax.text(0, 0, "EGO", color="white", fontsize=8, ha="center", va="center",
            weight="bold", zorder=6)

    ego_vel = frame.get("ego_velocity_vector", [0, 0, 0])
    ego_speed = math.sqrt(sum(v * v for v in ego_vel[:2]))
    if ego_speed > 0.1:
        # Forward axis is +X (plotted up), so an Vx push points up.
        ax.annotate("", xy=(0, min(ego_vel[0], view_range * 0.4)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#457b9d", lw=2), zorder=4)

    objects = frame.get("detected_objects", [])
    for obj in objects:
        cx = obj.get("center_xyz", [0, 0, 0])
        # Forward (X) is plotted on the vertical axis, lateral (Y) on horizontal.
        # Flip Y so "left of ego" appears on the left of the plot.
        px, py = -cx[1], cx[0]
        color = _class_color(obj.get("classification", ""))
        occ = obj.get("occlusion_state", "none")
        alpha = 0.35 if occ == "total" else (0.65 if occ and occ != "none" else 1.0)

        lwh = obj.get("extent_lwh", [4.0, 1.8, 1.5])
        ow, ol = lwh[1], lwh[0]
        ax.add_patch(Rectangle((px - ow / 2, py - ol / 2), ow, ol,
                               color=color, alpha=alpha, zorder=3,
                               linestyle="--" if occ == "total" else "-",
                               ec="black", lw=0.6))

        dist = math.sqrt(cx[0] ** 2 + cx[1] ** 2)
        label = f"{obj.get('classification', '?')} ({dist:.0f}m)"
        ax.text(px, py + ol / 2 + 1.0, label, color=color, fontsize=8,
                ha="center", va="bottom", weight="bold", zorder=4)

        vel = obj.get("velocity_vector", [0, 0, 0])
        if math.sqrt(vel[0] ** 2 + vel[1] ** 2) > 0.1:
            ax.annotate("", xy=(px - vel[1], py + vel[0]), xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=4)

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.set_xlabel("← left   lateral (m)   right →")
    ax.set_ylabel("← behind   forward (m)   ahead →")
    ax.axhline(0, color="#eeeeee", lw=0.5, zorder=0)
    ax.axvline(0, color="#eeeeee", lw=0.5, zorder=0)
    ax.set_title(f"t = {frame.get('frame_timestamp', 0):.2f}s   |   "
                 f"ego {ego_speed:.1f} m/s   |   {len(objects)} object(s)")
    fig.tight_layout()
    return fig

st.title("🛰️ Spatial Telemetry & Scenario Retrieval Engine")
st.caption("Local 3D Scene Discovery, Volumetric Processing & State Seeding Dashboard")

# --- Sidebar Telemetry Verification Monitoring Room ---
with st.sidebar:
    st.header("🛡️ Infrastructure Status")
    try:
        stats_res = requests.get(f"{API_BASE}/db/stats")
        if stats_res.status_code == 200:
            frame_count = stats_res.json().get("frame_count", 0)
            if frame_count == 0:
                st.success("🔄 DB Status: FRESHLY PURGED (0 Frames)")
            else:
                st.info(f"📊 DB Status: ACTIVE ({frame_count} Frames Seeded)")
        else:
            st.error("DB Synchronization Blocked")
    except Exception:
        st.warning("Backend API Daemon Offline")

    st.markdown("---")
    if st.button("🗑️ Clear Database", type="primary", width="stretch"):
        try:
            clear_res = requests.delete(f"{API_BASE}/db/clear")
            if clear_res.status_code == 200:
                st.success("Database cleared. All frames purged.")
                st.rerun()
            else:
                st.error(f"Clear failed: {clear_res.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.markdown("---")
    st.subheader("🎭 Act 1 Ingestion Quick Loaders")
    
    # Pre-baked Demo Playloads
    nuscenes_macro = {
        "source_dataset": "nuscenes",
        "scene_id": "nuscenes-log-segment-081",
        "frame_timestamp": 44.102,
        "ego_translation": [1042.4, 892.1, 12.3],
        "ego_velocity_vector": [8.4, 1.2, 0.0],
        "target_classification": "truck",
        "target_id": 4022,
        "box_translation": [1054.1, 896.7, 12.5],
        "box_size_lwh": [6.5, 2.2, 3.1],
        "orientation_parameter": [0.7071, 0.0, 0.0, 0.7071],
        "target_velocity_vector": [-2.1, 0.5, 0.0],
        "occlusion_state": "none"
    }

    waymo_macro = {
        "source_dataset": "waymo",
        "scene_id": "waymo-sf-mission-seq-1092",
        "frame_timestamp": 182.904,
        "ego_translation": [145.28, -212.14, 4.52],
        "ego_velocity_vector": [14.15, 0.22, -0.02],
        "target_classification": "motorcycle",
        "target_id": 771,
        "box_translation": [152.94, -222.31, 4.41],
        "box_size_lwh": [2.15, 0.85, 1.22],
        "orientation_parameter": [-1.5708],
        "target_velocity_vector": [12.44, -0.21, 0.0],
        "occlusion_state": "total"
    }

    if st.button("Load nuScenes (Truck) Payload"):
        st.session_state["bridge_json_payload"] = json.dumps(nuscenes_macro, indent=2)

    if st.button("Load Waymo (Motorcycle) Payload"):
        st.session_state["bridge_json_payload"] = json.dumps(waymo_macro, indent=2)

    st.markdown("---")
    st.subheader("🔎 Act 2 Query Injection Presets")
    if st.button("Load Query A: Motorcycle Search"):
        st.session_state["active_search_prompt"] = "Find an oncoming or nearby motorcycle or bike with extreme obstruction or occlusion"

    if st.button("Load Query B: Truck Search"):
        st.session_state["active_search_prompt"] = "Retrieve a large commercial vehicle or heavy freight truck driving far away with completely clear line of sight"

    if st.button("Load Query C: Edge Case Robustness Check"):
        st.session_state["active_search_prompt"] = "Show me a tailgating electric autonomous shuttle or delivery drone in heavy rain"

# --- Scene Viewer (Top-Down BEV Visualizer) ---
st.header("🗺️ Scene Viewer — Top-Down BEV")
st.caption("Visualize what a scene actually looks like before you query it. Ego vehicle is the dark box at center, facing up (+X forward). Detected objects are color-coded by class; faded/dashed = occluded.")

try:
    scenes_res = requests.get(f"{API_BASE}/scenes", timeout=5)
    scene_list = scenes_res.json().get("scenes", []) if scenes_res.status_code == 200 else []
except Exception:
    scene_list = []

scene_ids = [s["scene_id"] for s in scene_list]

if not scene_list:
    st.info("No scenes in the database yet. Ingest some frames (or run seed_test_data.py) to populate the viewer.")
else:
    scene_labels = [f"{s['scene_id']}  ({s['frame_count']} frames)" for s in scene_list]

    # If a "View in BEV" jump was requested from the scenario panel, apply it
    # here BEFORE the selectbox renders so the widget picks up the new value.
    jump_target = st.session_state.pop("bev_jump_target", None)
    if jump_target and jump_target in scene_ids:
        st.session_state["bev_scene_select"] = scene_ids.index(jump_target)
        st.session_state["bev_loaded_scene"] = None
        st.session_state["bev_frame_idx"] = 0

    ctrl_a, ctrl_b = st.columns([3, 2])
    with ctrl_a:
        selected_idx = st.selectbox(
            "Scene:", range(len(scene_ids)),
            format_func=lambda i: scene_labels[i], key="bev_scene_select"
        )
    selected_scene = scene_ids[selected_idx]

    # Fetch frames for the selected scene (cached in session per scene).
    cache_key = f"bev_frames::{selected_scene}"
    if st.session_state.get("bev_loaded_scene") != selected_scene:
        try:
            fr = requests.get(f"{API_BASE}/frames/{selected_scene}", timeout=10)
            st.session_state[cache_key] = fr.json().get("frames", []) if fr.status_code == 200 else []
        except Exception as e:
            st.session_state[cache_key] = []
            st.error(f"Could not load frames: {e}")
        st.session_state["bev_loaded_scene"] = selected_scene
        st.session_state["bev_frame_idx"] = 0

    frames = st.session_state.get(cache_key, [])

    if not frames:
        st.warning("No frames returned for this scene.")
    else:
        n = len(frames)

        with ctrl_b:
            auto = st.checkbox("⏵ Auto-play", key="bev_autoplay")
            play_speed = st.slider("Frame delay (s)", 0.1, 2.0, 0.5, 0.1, key="bev_speed")

        # bev_frame_idx is the single source of truth — buttons and autoplay
        # mutate it before the slider renders, slider reads it via value=.
        # Never write to a slider's own key= after the widget is instantiated.
        st.session_state.setdefault("bev_frame_idx", 0)
        st.session_state["bev_frame_idx"] = min(st.session_state["bev_frame_idx"], n - 1)

        nav_prev, nav_slider, nav_next = st.columns([1, 4, 1])
        with nav_prev:
            if st.button("◀ Prev", width="stretch", key="bev_prev"):
                st.session_state["bev_frame_idx"] = max(0, st.session_state["bev_frame_idx"] - 1)
        with nav_next:
            if st.button("Next ▶", width="stretch", key="bev_next"):
                st.session_state["bev_frame_idx"] = min(n - 1, st.session_state["bev_frame_idx"] + 1)
        with nav_slider:
            # No key= — capture return value directly so Streamlit never owns
            # this slot and we can freely update bev_frame_idx from anywhere.
            dragged = st.slider("Frame", 0, n - 1, value=st.session_state["bev_frame_idx"])
            st.session_state["bev_frame_idx"] = dragged

        idx = st.session_state["bev_frame_idx"]
        frame = frames[idx]

        plot_col, table_col = st.columns([3, 2])
        with plot_col:
            st.caption(f"**Scene:** `{selected_scene}`  |  **Frame:** {idx + 1} / {n}  |  **t = {frame.get('frame_timestamp', 0):.3f}s**")
            st.pyplot(render_bev_frame(frame))
        with table_col:
            st.markdown("**Objects in this frame:**")
            objs = frame.get("detected_objects", [])
            if objs:
                table_rows = []
                for o in objs:
                    cx = o.get("center_xyz", [0, 0, 0])
                    vel = o.get("velocity_vector", [0, 0, 0])
                    table_rows.append({
                        "class": o.get("classification", "?"),
                        "dist (m)": round(math.sqrt(cx[0] ** 2 + cx[1] ** 2), 1),
                        "X": round(cx[0], 1), "Y": round(cx[1], 1),
                        "Vx": round(vel[0], 1), "Vy": round(vel[1], 1),
                        "occlusion": o.get("occlusion_state", "none"),
                    })
                st.dataframe(table_rows, width="stretch", hide_index=True)
            else:
                st.info("No objects detected in this frame (empty scene).")

        # Drive auto-play: advance bev_frame_idx then rerun after a short pause.
        if st.session_state.get("bev_autoplay"):
            import time as _time
            st.session_state["bev_frame_idx"] = (idx + 1) % n
            _time.sleep(st.session_state.get("bev_speed", 0.5))
            st.rerun()

st.markdown("---")

# --- Workspaces Processing Layout Matrix ---
col1, col2 = st.columns(2)

with col1:
    st.header("📥 Ingestion Workspace")
    tab1, tab2 = st.tabs(["Native Stream Payload", "🛠️ Real-World Dataset Bridge"])
    
    with tab1:
        st.subheader("Stream Telemetry Configuration Package")
        sample_payload = {
            "scene_id": "miata-sequence-042",
            "frame_timestamp": 12.450,
            "ego_velocity_vector": [11.2, 0.0, -0.05],
            "camera_extrinsics_rt": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.2], [0.0, 0.0, 1.0, -0.4]],
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
        native_json_input = st.text_area("Structured Spatial JSON Input Contract:", value=json.dumps(sample_payload, indent=2), height=350)
        
        if st.button("🚀 Push Telemetry to Database Cluster", key="btn_native_push"):
            try:
                parsed_payload = json.loads(native_json_input)
                with st.spinner("Processing framework telemetry injection..."):
                    res = requests.post(f"{API_BASE}/ingest/spatial", json=parsed_payload)
                if res.status_code in [200, 202]:
                    st.success(f"Successfully queued sequence via background queue pipelines.")
                else:
                    st.error(f"Backend Server Error Flag: {res.text}")
            except Exception as e:
                st.error(f"Transport Connection Failed: {e}")

    with tab2:
        st.subheader("Open-Source Autonomous Tracking Transformer")
        
        # Verify Session State Hydration for Macros
        if "bridge_json_payload" not in st.session_state:
            st.session_state["bridge_json_payload"] = json.dumps(nuscenes_macro, indent=2)

        bridge_json_input = st.text_area(
            "Bridge Input Struct:", 
            value=st.session_state["bridge_json_payload"], 
            height=350, 
            key="bridge_text_area"
        )

        if st.button("⚡ Transform & Inject Open Matrix", key="btn_bridge_push"):
            try:
                parsed_bridge_payload = json.loads(bridge_json_input)
                with st.spinner("Executing coordinate transformations and vector pipeline handoff..."):
                    response = requests.post(f"{API_BASE}/ingest/dataset-bridge", json=parsed_bridge_payload)
                
                # Check for 202 ACCEPTED alongside standard 200
                if response.status_code in [200, 202]:
                    st.success("Bridge Success: Unified and stored relative tracking entry inside database layer.")
                    st.rerun()
                else:
                    st.error(f"Bridge Transformation Exception: {response.text}")
            except json.JSONDecodeError:
                st.error("Invalid Configuration Structure (Malformed JSON)")
            except Exception as e:
                st.error(f"Handoff Connection Refused: {e}")

with col2:
    st.header("🔍 Query Workspace")
    sql_tab, scenario_tab = st.tabs(["🗄️ NL → SQL Query", "🤖 Scenario Vector Search"])

    with scenario_tab:
        st.subheader("Scenario Matrix Constraint Extraction")

        if "active_search_prompt" not in st.session_state:
            st.session_state["active_search_prompt"] = ""

        query_prompt = st.text_input(
            "Search Query:",
            placeholder="e.g., Find oncoming vehicles under occlusion closer than 15 meters",
            key="active_search_prompt"
        )

        if st.button("⚡ Run Spatial Search Strategy", key="btn_search_execute"):
            if query_prompt:
                with st.spinner("Computing prompt cosine distance matrices..."):
                    try:
                        response = requests.post(f"{API_BASE}/query/scenario", json={"prompt": query_prompt})
                        if response.status_code == 200:
                            res_data = response.json()
                            answer = res_data.get("answer", "No response payload.")
                            status_flag = res_data.get("status", "UNKNOWN")
                            vector_seed = res_data.get("state_vector_seed", None)
                            source_frames = res_data.get("source_frames", [])

                            # --- Source attribution --- show which DB record
                            # the LLM was given so the user can verify the output.
                            if source_frames:
                                st.markdown("**Retrieved from database:**")
                                for sf in source_frames:
                                    sid = sf["scene_id"]
                                    ts  = sf["frame_timestamp"]
                                    dist = sf.get("similarity_distance", 0)
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.info(
                                            f"Scene: `{sid}`  |  t = {ts:.3f}s  |  "
                                            f"similarity distance: {dist:.3f}"
                                        )
                                    with col_b:
                                        if st.button(
                                            "View in BEV ↑",
                                            key=f"jump_{sid}_{ts}",
                                            width="stretch",
                                        ):
                                            # Write to bev_jump_target, NOT bev_scene_select.
                                            # The selectbox owns its own key after render so
                                            # we can't write to it here — instead the Scene
                                            # Viewer reads bev_jump_target before it renders
                                            # and applies it to bev_scene_select in time.
                                            st.session_state["bev_jump_target"] = sid
                                            st.rerun()

                            st.markdown("---")
                            st.subheader("🤖 Synthesized Engineering Output:")
                            st.markdown(answer)

                            st.markdown("---")
                            st.subheader("🔢 Parsed State Vector")
                            st.caption(
                                "These numbers are extracted from the LLM output above. "
                                "Cross-check them against the 'Retrieved from database' frame above — "
                                "the LLM can occasionally alter values in its response."
                            )
                            st.metric(label="Extraction Status", value=status_flag)
                            st.write("$x = [X, Y, V_x, V_y]^T$:", vector_seed)
                        else:
                            st.error(f"Backend Query Processing Exception: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with API server: {e}")
            else:
                st.warning("Please specify a constraint scenario query sequence first.")

    with sql_tab:
        st.subheader("Natural Language → SQL Generator")
        st.caption("Describe what you want to retrieve in plain English. The engine will generate, validate, and execute a SQL query against the spatial telemetry store.")

        if "sql_nl_input_field" not in st.session_state:
            st.session_state["sql_nl_input_field"] = ""

        sql_query_presets = {
            "Show all frames": "Show me all ingested scene frames with their timestamps and scene IDs",
            "Totally occluded objects": "Find all frames containing objects with total occlusion",
            "Fast ego frames": "Show frames where the ego vehicle was traveling faster than 10 m/s",
            "Count by scene": "Count how many frames exist per scene ID",
        }

        st.markdown("**Quick Presets:**")
        preset_cols = st.columns(2)
        for i, (label, preset_prompt) in enumerate(sql_query_presets.items()):
            with preset_cols[i % 2]:
                if st.button(label, key=f"sql_preset_{i}"):
                    st.session_state["sql_nl_input_field"] = preset_prompt

        sql_nl_input = st.text_input(
            "Natural Language Query:",
            placeholder="e.g., Show all frames with motorcycles that have total occlusion",
            key="sql_nl_input_field"
        )

        if st.button("⚡ Generate & Execute SQL", key="btn_sql_execute"):
            if sql_nl_input:
                with st.spinner("Generating SQL via llama3.2 and executing against spatial store..."):
                    try:
                        response = requests.post(f"{API_BASE}/query/sql", json={"prompt": sql_nl_input})
                        if response.status_code == 200:
                            res_data = response.json()
                            generated_sql = res_data.get("generated_sql", "")
                            rows = res_data.get("rows", [])
                            row_count = res_data.get("row_count", 0)
                            status_flag = res_data.get("status", "UNKNOWN")
                            detail = res_data.get("detail", "")

                            st.markdown("**Generated SQL:**")
                            st.code(generated_sql, language="sql")

                            st.metric(label="Execution Status", value=status_flag)

                            if status_flag == "SUCCESS":
                                st.caption(f"{row_count} row(s) returned")
                                if rows:
                                    st.dataframe(rows, width="stretch")
                                else:
                                    st.info("Query executed successfully — no rows matched.")
                            else:
                                st.error(f"Detail: {detail}")
                        else:
                            st.error(f"Backend Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            else:
                st.warning("Enter a natural language query first.")
