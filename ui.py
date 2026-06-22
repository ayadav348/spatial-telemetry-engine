import streamlit as st
import requests
import json

st.set_page_config(layout="wide", page_title="Spatial Telemetry Dashboard")

API_BASE = "http://127.0.0.1:8000"

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
    if st.button("🗑️ Clear Database", type="primary", use_container_width=True):
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

                            st.subheader("🤖 Synthesized Engineering Output:")
                            st.markdown(answer)

                            st.markdown("---")
                            st.subheader("🔢 Downstream Tracking Metrics Registry")
                            st.metric(label="Extraction Status Profile", value=status_flag)
                            st.write("Parsed Target Matrix Seed ($x$):", vector_seed)
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
                                    st.dataframe(rows, use_container_width=True)
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
