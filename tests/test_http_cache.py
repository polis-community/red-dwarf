from io import BytesIO

from requests import Request, Response
from urllib3.response import HTTPResponse

from reddwarf.helpers import CachedLimiterSession


def test_cached_limiter_session_round_trips_response_with_cookies(tmp_path):
    response = Response()
    response.status_code = 200
    response.url = "https://example.test/data"
    response.request = Request("GET", response.url).prepare()
    response.raw = HTTPResponse(
        body=BytesIO(b'{"ok":true}'),
        preload_content=False,
        request_url=response.url,
    )
    response.cookies.set("test_cookie", "test_value")
    assert response.content == b'{"ok":true}'

    with CachedLimiterSession(cache_name=str(tmp_path / "http_cache")) as session:
        session.cache.save_response(response)
        cache_key = session.cache.create_key(response.request)

    with CachedLimiterSession(cache_name=str(tmp_path / "http_cache")) as session:
        cached_response = session.cache.get_response(cache_key)
        assert cached_response is not None
        assert cached_response.from_cache is True
        assert cached_response.json() == {"ok": True}
        assert cached_response.cookies.get("test_cookie") == "test_value"
