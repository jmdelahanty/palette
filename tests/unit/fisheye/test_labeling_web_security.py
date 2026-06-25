from __future__ import annotations

from fisheye.labeling import web as labeling_web


class _Handler:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def test_same_origin_allows_requests_without_browser_origin_headers():
    assert labeling_web._request_has_same_origin(_Handler({}))
    assert labeling_web._request_has_same_origin(_Handler({"Host": "labeling.example.org"}))


def test_same_origin_accepts_matching_origin_or_referer():
    assert labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Origin": "https://labeling.example.org"})
    )
    assert labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Referer": "https://labeling.example.org/r/session-a"})
    )
    assert labeling_web._request_has_same_origin(
        _Handler({"Host": "127.0.0.1:8795", "Origin": "http://127.0.0.1:8795"})
    )


def test_same_origin_rejects_mismatched_origin_or_referer():
    assert not labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Origin": "https://evil.example.org"})
    )
    assert not labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Referer": "https://evil.example.org/r/session-a"})
    )
    assert not labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Origin": "null"})
    )


def test_same_origin_uses_forwarded_host_when_present():
    assert labeling_web._request_has_same_origin(
        _Handler(
            {
                "Host": "127.0.0.1:8795",
                "X-Forwarded-Host": "labeling.example.org",
                "Origin": "https://labeling.example.org",
            }
        )
    )
    assert not labeling_web._request_has_same_origin(
        _Handler(
            {
                "Host": "127.0.0.1:8795",
                "X-Forwarded-Host": "labeling.example.org",
                "Origin": "https://127.0.0.1:8795",
            }
        )
    )
