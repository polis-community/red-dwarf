from requests import Session
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin
from reddwarf.requests.ip_rotator_mixin import IPRotatorMixin


class RedDwarfSession(CacheMixin, LimiterMixin, IPRotatorMixin, Session):
    """
    Session class with optional caching, rate-limiting, and IP rotation behavior. Accepts arguments for both
    LimiterSession and CachedSession.

    IP rotation is done via AWS gateway.

    See: https://requests-cache.readthedocs.io/en/stable/user_guide/compatibility.html#requests-ratelimiter
    See: https://github.com/Ge0rg3/requests-ip-rotator
    """