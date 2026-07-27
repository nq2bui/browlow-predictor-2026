import time
from urllib.parse import urlparse

import requests

_USER_AGENT = "brownlow-predictor-2026/1.0 (+https://github.com/nq2bui/browlow-predictor-2026)"
_last_request_time: dict[str, float] = {}


def fetch_url(url: str, min_interval_seconds: float = 1.0) -> str:
    host = urlparse(url).netloc
    last = _last_request_time.get(host)
    if last is not None:
        elapsed = time.time() - last
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    response.raise_for_status()
    _last_request_time[host] = time.time()
    return response.text
