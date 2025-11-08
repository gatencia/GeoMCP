import asyncio
import base64
import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

OUTPUT_DIR = Path(__file__).resolve().parent / "matrix_file_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_TOOLS = {
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
}


def _unique_path(tool_name: str, suffix: str) -> Path:
    safe_name = tool_name.replace("/", "_")
    candidate = OUTPUT_DIR / f"{safe_name}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = OUTPUT_DIR / f"{safe_name}_{counter}{suffix}"
        counter += 1
    return candidate


def _load_structured_payload(result: object) -> dict | None:
    for attr in (
        "structured_output",
        "structuredOutput",
        "structured_content",
        "structuredContent",
    ):
        structured = getattr(result, attr, None)
        if structured is not None:
            return structured

    for method in ("model_dump", "dict"):
        func = getattr(result, method, None)
        if func is None:
            continue
        try:
            dumped = func()
        except TypeError:
            try:
                dumped = func(exclude_none=True)
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(dumped, dict):
            for key in (
                "structured_output",
                "structuredOutput",
                "structured_content",
                "structuredContent",
            ):
                if key in dumped:
                    return dumped[key]
    meta = getattr(result, "meta", None)
    if isinstance(meta, dict):
        for key in (
            "structured_output",
            "structuredOutput",
            "structured_content",
            "structuredContent",
        ):
            if key in meta:
                return meta[key]
    return None


def _save_resource_block(tool_name: str, block: object) -> Path | None:
    resource = getattr(block, "resource", None)
    if resource is None:
        return None

    uri = getattr(resource, "uri", None)
    suffix = ""
    if uri:
        uri_str = str(uri)
        parsed = urlparse(uri_str)
        suffix = Path(parsed.path).suffix
    if not suffix:
        suffix = ".bin"

    if hasattr(resource, "blob") and resource.blob:
        try:
            data = base64.b64decode(resource.blob)
        except Exception:
            data = None
        if data is not None:
            dest = _unique_path(tool_name, suffix)
            dest.write_bytes(data)
            return dest

    if uri:
        parsed = urlparse(str(uri))
        if parsed.scheme == "file":
            src_path = Path(unquote(parsed.path))
            if src_path.exists():
                dest = _unique_path(tool_name, suffix)
                shutil.copy2(src_path, dest)
                return dest

    if hasattr(resource, "text") and resource.text is not None:
        dest = _unique_path(tool_name, suffix if suffix != ".bin" else ".txt")
        dest.write_text(resource.text)
        return dest

    return None


def _save_result(tool: str, result) -> tuple[Path | None, Path | None]:
    saved_data = None
    metadata_path = None

    if getattr(result, "content", None):
        for block in result.content:
            saved_data = _save_resource_block(tool, block)
            if saved_data:
                break

    structured = _load_structured_payload(result)
    if structured is not None:
        metadata_path = _unique_path(tool, "_metadata.json")
        metadata_path.write_text(json.dumps(structured, indent=2, default=str))

    return saved_data, metadata_path


async def main() -> None:
    async with Client(StreamableHttpTransport("http://127.0.0.1:8765/mcp")) as client:
        for tool, params in MATRIX_TOOLS.items():
            args = dict(params)
            args["as_file"] = True
            try:
                result = await client.call_tool(tool, args)
                saved_file, metadata_file = _save_result(tool, result)
                if saved_file:
                    info = f"file={saved_file.name}"
                else:
                    info = "no file"
                if metadata_file:
                    info += f", metadata={metadata_file.name}"
                print(f"{tool:28} PASS -> {info}")
            except Exception as exc:
                print(f"{tool:28} FAIL -> {exc}")


if __name__ == "__main__":
    asyncio.run(main())
