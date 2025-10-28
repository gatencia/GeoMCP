import time
from modules.sentinel_hub import _token_cache, _get_token

def get_status():
    token = _get_token()
    exp = _token_cache.get("expires_at", 0)
    ttl = max(0, exp - time.time())
    return {
        "token_cached": bool(token),
        "expires_in_sec": round(ttl, 1),
        "minutes_remaining": round(ttl / 60, 1),
        "active": ttl > 0,
    }