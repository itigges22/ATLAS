import os
from pydantic import BaseModel


class ServerConfig(BaseModel):
    port: int = 8099
    host: str = "0.0.0.0"


class LlamaConfig(BaseModel):
    base_url: str = os.environ.get("LLAMA_URL", "http://llama-server:8080")


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    llama: LlamaConfig = LlamaConfig()


config = Config()


def online_learning_enabled() -> bool:
    """Whether the pattern cache may change while the service is running.

    On by default: the cache learns from what it serves, which is the point
    of it. ATLAS_LENS_ONLINE_LEARNING=0 freezes it -- patterns are still
    retrieved and served, and lens scoring is untouched, but nothing writes
    back.

    A controlled paired run needs this. Three things mutate the store, and
    only the first is obvious: writes add patterns; reads update
    last_accessed and access_count, which retrieval scores on; and the read
    path bumps hit/miss counters synchronously. Without a freeze the
    patterns an early case touches change what a later case is served, so
    the two arms are no longer being run against the same cache and case
    order becomes a variable in the result.

    Read per call rather than cached at import so a test can set it without
    reimporting the module tree.
    """
    raw = os.environ.get("ATLAS_LENS_ONLINE_LEARNING", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
