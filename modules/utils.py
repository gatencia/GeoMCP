def bbox_from_query(bbox_str: str):
    """
    Convert a bbox query string 'minLon,minLat,maxLon,maxLat'
    into a list of floats [minLon, minLat, maxLon, maxLat].
    """
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError("Bounding box must have 4 comma-separated values.")
        return parts
    except Exception as e:
        raise ValueError(f"Invalid bbox format: {e}")