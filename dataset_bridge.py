import numpy as np
import httpx
import asyncio

class DatasetTelemetryBridge:
    def __init__(self, fastapi_endpoint: str = "http://localhost:8000/ingest"):
        self.endpoint = fastapi_endpoint

    def extract_nuscenes_kinematics(self, translation: list, rotation_quad: list, velocity_2d: list) -> dict:
        """
        Transforms nuScenes coordinate frames. Converts orientation quaternions 
        into continuous yaw angles and calculates total velocity magnitude.
        """
        vx, vy = velocity_2d[0], velocity_2d[1]
        if np.isnan(vx) or np.isnan(vy):
            vx, vy = 0.0, 0.0
            
        # Extract yaw angle from Quaternion [w, x, y, z]
        w, x, y, z = rotation_quad
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        v_magnitude = float(np.sqrt(vx**2 + vy**2))
        
        return {
            "matrix_type": "nuScenes",
            "coords": {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2])},
            "yaw": float(yaw),
            "vx": float(vx),
            "vy": float(vy),
            "v_magnitude": v_magnitude
        }

    def extract_waymo_kinematics(self, center_xyz: list, heading_yaw: float, speed_xy: list) -> dict:
        """
        Parses Waymo Open Dataset bounding boxes from coordinate frames.
        """
        vx, vy = speed_xy[0], speed_xy[1]
        v_magnitude = float(np.sqrt(vx**2 + vy**2))
        
        return {
            "matrix_type": "Waymo",
            "coords": {"x": float(center_xyz[0]), "y": float(center_xyz[1]), "z": float(center_xyz[2])},
            "yaw": float(heading_yaw),
            "vx": float(vx),
            "vy": float(vy),
            "v_magnitude": v_magnitude
        }

    async def pipe_to_engine(self, payload: dict):
        """Pipes the normalized spatial matrix straight into your FastAPI backend."""
        async with httpx.AsyncClient() as client:
            # Matches your Thursday structural code layouts
            formatted_payload = {
                "telemetry": {
                    "velocity_magnitude": payload["v_magnitude"],
                    "vx": payload["vx"],
                    "vy": payload["vy"]
                },
                "bounding_box": {
                    "x": payload["coords"]["x"],
                    "y": payload["coords"]["y"],
                    "z": payload["coords"]["z"],
                    "yaw": payload["yaw"]
                }
            }
            try:
                response = await client.post(f"{self.endpoint}", json=formatted_payload, timeout=2.0)
                return response.status_code
            except Exception as e:
                return f"Connection_Error: {str(e)}"
