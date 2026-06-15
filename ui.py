import streamlit as st
import requests
import json

st.set_page_config(layout="wide", page_title="Spatial Telemetry Dashboard")

API_BASE = "http://127.0.0.1:8000"

st.title("🛰️ Spatial Telemetry & Scenario Retrieval Engine")
st.caption("Local 3D Scene Discovery, Volumetric Processing & State Seeding Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("📥 Ingestion Workspace")
    
    # Establish separate tabs within Ingestion to support raw native streaming vs dataset translation bridges
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
                },
                {
                    "target_id": 106,
                    "classification": "pedestrian",
                    "center_xyz": [-1.8, 8.5, -0.2],
                    "extent_lwh": [0.6, 0.6, 1.7],
                    "velocity_vector": [0.0, 1.1, 0.0],
                    "occlusion_state": "none"
                }
            ]
        }
        
        json_input = st.text_area(
            "Structured Spatial JSON Input Contract:",
            value=json.dumps(sample_payload, indent=2),
            height=350,
            key="native_json_input"
        )
        
        if st.button("🚀 Push Telemetry to Database Cluster", key="btn_native_push"):
            try:
                parsed_payload = json.loads(json_input)
                with st.spinner("Executing matrix serialization and pgvector indexing..."):
                    response = requests.post(f"{API_BASE}/ingest/spatial", json=parsed_payload)
                    
                if response.status_code == 200:
                    res_data = response.json()
                    st.success(f"Successfully indexed sequence: {res_data['scene_indexed']} at timestamp {res_data['frame_timestamp']}s")
                else:
                    st.error(f"Backend Server Error Flag: {response.text}")
            except json.JSONDecodeError:
                st.error("Data Verification Failure: Input block is not a valid JSON string structure.")
            except Exception as e:
                st.error(f"Transport Connection Failed: {e}")

    with tab2:
        st.subheader("Open-Source Autonomous Tracking Transformer")
        st.caption("Normalizes tracking logs from heterogeneous data layers straight to your local cluster rules.")
        
        dataset_mode = st.selectbox("Target Telemetry Protocol Frame", ["nuScenes (Quaternion-Based)", "Waymo (Yaw-Based)"])
        
        # Hydrate mock configurations based on target framework parameters
        if "nuScenes" in dataset_mode:
            bridge_payload = {
                "source_dataset": "nuscenes",
                "scene_id": "nuscenes-log-segment-081",
                "frame_timestamp": 44.102,
                "ego_translation": [1042.4, 892.1, 12.3],
                "ego_velocity_vector": [8.4, 1.2, 0.0],
                "target_classification": "truck",
                "target_id": 4022,
                "box_translation": [1054.1, 896.7, 12.5],
                "box_size_lwh": [6.5, 2.2, 3.1],
                "orientation_parameter": [0.707, 0.0, 0.0, 0.707], # [w, x, y, z] Matrix Orientation 
                "target_velocity_vector": [-2.1, 0.5, 0.0],
                "occlusion_state": "none"
            }
        else:
            bridge_payload = {
                "source_dataset": "waymo",
                "scene_id": "waymo-run-segment-119",
                "frame_timestamp": 182.904,
                "ego_translation": [45.2, -12.1, 0.5],
                "ego_velocity_vector": [14.1, 0.0, -0.02],
                "target_classification": "motorcycle",
                "target_id": 771,
                "box_translation": [52.9, -10.3, 0.4],
                "box_size_lwh": [2.1, 0.8, 1.2],
                "orientation_parameter": [1.5708], # Raw heading angle scalar wrapped in array
                "target_velocity_vector": [12.4, -0.2, 0.0],
                "occlusion_state": "total"
            }

        bridge_json_input = st.text_area(
            "Bridge Input Struct:",
            value=json.dumps(bridge_payload, indent=2),
            height=350,
            key="bridge_json_input"
        )

        if st.button("⚡ Transform & Inject Open Matrix", key="btn_bridge_push"):
            try:
                parsed_bridge_payload = json.loads(bridge_json_input)
                with st.spinner("Processing coordinate transformations and executing vector pipeline handoff..."):
                    response = requests.post(f"{API_BASE}/ingest/dataset-bridge", json=parsed_bridge_payload)
                    
                if response.status_code == 200:
                    res_data = response.json()
                    st.success(f"Bridge Success: Unified and stored track entry inside database.")
                else:
                    st.error(f"Bridge Transformation Exception: {response.text}")
            except json.JSONDecodeError:
                st.error("Invalid Configuration Structure (Malformed JSON)")
            except Exception as e:
                st.error(f"Handoff Connection Refused: {e}")

with col2:
    st.header("🔍 Conversational Query Workspace")
    st.subheader("Scenario Matrix Constraint Extraction")
    
    query_prompt = st.text_input(
        "Search Query:",
        placeholder="e.g., Find oncoming vehicles under occlusion closer than 15 meters",
        key="txt_search_query"
    )
    
    if st.button("⚡ Run Spatial Search Strategy", key="btn_search_execute"):
        if query_prompt:
            with st.spinner("Computing query embeddings and scanning cosine distance matrices..."):
                try:
                    response = requests.post(f"{API_BASE}/query/scenario", json={"prompt": query_prompt})
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No response content returned.")
                        st.subheader("🤖 Synthesized Engineering Output:")
                        st.markdown(answer)
                    else:
                        st.error(f"Backend Query Processing Exception: {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API server: {e}")
        else:
            st.warning("Please specify a constraint scenario query sequence first.")
