import time
from unittest.mock import patch, MagicMock
from brownlow.http import fetch_url


def test_fetch_url_returns_response_text():
    mock_response = MagicMock()
    mock_response.text = "<html>ok</html>"
    mock_response.raise_for_status = MagicMock()
    with patch("brownlow.http.requests.get", return_value=mock_response) as mock_get:
        result = fetch_url("https://example.com/page.html")
    assert result == "<html>ok</html>"
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["headers"]["User-Agent"]


def test_fetch_url_rate_limits_same_host():
    mock_response = MagicMock()
    mock_response.text = "<html>ok</html>"
    mock_response.raise_for_status = MagicMock()
    with patch("brownlow.http.requests.get", return_value=mock_response):
        with patch("brownlow.http.time.sleep") as mock_sleep:
            fetch_url("https://example.com/a.html", min_interval_seconds=2.0)
            fetch_url("https://example.com/b.html", min_interval_seconds=2.0)
    mock_sleep.assert_called()
