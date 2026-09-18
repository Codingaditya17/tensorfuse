"""
remote_cache.py — thin client for tuning_cache_service (Go). Used by
autotune.py only when the TENSORFUSE_CACHE_URL environment variable is
set; otherwise autotune.py falls back to its local JSON file exactly
as before. No new dependency -- uses urllib from the standard library.
"""

import json
import os
import urllib.parse
import urllib.request
import urllib.error


def get_entry(base_url, sig, cols):
    url = f"{base_url}/tuning?sig={urllib.parse.quote(sig)}&cols={cols}"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None  # network hiccup -- caller falls back to a fresh search


def put_entry(base_url, sig, cols, entry: dict):
    url = f"{base_url}/tuning?sig={urllib.parse.quote(sig)}&cols={cols}"
    data = json.dumps(entry).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def cache_url():
    return os.environ.get("TENSORFUSE_CACHE_URL")
