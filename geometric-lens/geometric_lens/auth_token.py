"""Internal service-auth token (lens side).

Reads the per-installation token from ATLAS_SERVICE_TOKEN_FILE
(default /run/atlas-secrets/service-token, the read-only compose
mount). Empty => auth disabled: inbound requests are accepted and
outbound requests carry no header — the pre-token behavior. The token
value is never logged.

install_urllib_opener() covers every urllib.request call site in the
package (embedding extraction, identity probe) with one opener;
httpx call sites merge auth_headers() explicitly.
"""

import os
import urllib.request


def _load() -> str:
    path = os.environ.get("ATLAS_SERVICE_TOKEN_FILE",
                          "/run/atlas-secrets/service-token")
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


SERVICE_TOKEN = _load()


def auth_headers() -> dict:
    if not SERVICE_TOKEN:
        return {}
    return {"Authorization": f"Bearer {SERVICE_TOKEN}"}


def install_urllib_opener() -> None:
    """Global opener so bare urlopen()/Request sites send the token.

    urllib merges addheaders under explicit per-request headers, so a
    request that sets its own Authorization keeps it.
    """
    if not SERVICE_TOKEN:
        return
    opener = urllib.request.build_opener()
    opener.addheaders = [("Authorization", f"Bearer {SERVICE_TOKEN}")]
    urllib.request.install_opener(opener)
