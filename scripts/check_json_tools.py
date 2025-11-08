import asyncio
import json
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

TOOLS = {
    "health": {},
    "list_capabilities": {},
    "get_ndwi_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "from_date": "2024-05-01",
        "to_date": "2024-05-10",
        "width": 32,
        "height": 32,
        "force_http": False,
    },
    "get_ndbi_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "from_date": "2024-05-01",
        "to_date": "2024-05-10",
        "width": 32,
        "height": 32,
        "force_http": False,
    },
    "get_elevation_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "width": 32,
        "height": 32,
        "force_http": False,
    },
    "get_slope_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "width": 32,
        "height": 32,
        "force_http": False,
    },
    "get_aspect_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "width": 32,
        "height": 32,
        "force_http": False,
    },
    "get_hillshade_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "width": 32,
        "height": 32,
        "azimuth_deg": 315.0,
        "altitude_deg": 45.0,
        "force_http": False,
    },
    "get_flow_accumulation_matrix": {
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "width": 32,
        "height": 32,
        "force_http": False,
    },
    "get_zonal_timeseries_json": {
        "index": "NDVI",
        "from_date": "2024-05-01",
        "to_date": "2024-05-10",
        "step_days": 5,
        "geometry": None,
        "bbox": [-74.06, 40.67, -73.86, 40.81],
        "force_http": False,
    },
    "get_point_timeseries_json": {
        "index": "NDVI",
        "from_date": "2024-05-01",
        "to_date": "2024-05-10",
        "step_days": 5,
        "lat": 40.74,
        "lon": -73.95,
        "force_http": False,
    },
}

async def main():
    async with Client(StreamableHttpTransport("http://127.0.0.1:8765/mcp")) as client:
        for tool, params in TOOLS.items():
            try:
                result = await client.call_tool(tool, params)
                payload = None
                if result.content:
                    block = result.content[0]
                    if hasattr(block, "data"):
                        payload = block.data
                    elif hasattr(block, "text"):
                        try:
                            payload = json.loads(block.text)
                        except json.JSONDecodeError:
                            payload = block.text
                status = "PASS" if isinstance(payload, (dict, list)) else "BAD_PAYLOAD"
                print(f"{tool:28} {status}")
            except Exception as exc:
                print(f"{tool:28} FAIL -> {exc}")

asyncio.run(main())