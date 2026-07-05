"""Sign and verify lens/ASA bundle manifests.

Builds on the per-bundle provenance.json (geometric_lens.provenance):
a bundle can carry an SSH signature over its manifest, and verification
checks BOTH the signature (against .github/allowed_signers) AND that
every file's on-disk SHA-256 still matches the manifest. So a tampered
artifact fails even if the manifest is intact, and a swapped manifest
fails the signature — trust needs both integrity and a known signer.

Uses the same SSH signing key as release tags (git config
user.signingkey); no new key material. Signing needs the private key
(maintainer machine); verification needs only allowed_signers.
"""

import hashlib
import os
import subprocess
from typing import List, Optional, Tuple

MANIFEST = "provenance.json"
SIGNATURE = "provenance.json.sig"
NAMESPACE = "atlas-artifact"


def _repo_root() -> str:
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, "docker-compose.yml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd()


def _allowed_signers() -> str:
    return os.path.join(_repo_root(), ".github", "allowed_signers")


def _signing_key() -> Optional[str]:
    try:
        key = subprocess.check_output(
            ["git", "config", "--get", "user.signingkey"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return key or None
    except (subprocess.SubprocessError, OSError):
        return None


def sign_manifest(bundle_dir: str) -> str:
    """Sign provenance.json with the configured SSH signing key. Returns
    the signature path. Raises RuntimeError if signing isn't set up."""
    key = _signing_key()
    if not key:
        raise RuntimeError(
            "no SSH signing key configured (git config user.signingkey); "
            "see docs/RELEASE.md signing setup")
    manifest = os.path.join(bundle_dir, MANIFEST)
    if not os.path.isfile(manifest):
        raise RuntimeError(f"no {MANIFEST} in {bundle_dir}")
    subprocess.check_call(
        ["ssh-keygen", "-Y", "sign", "-f", key, "-n", NAMESPACE, manifest],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(bundle_dir, SIGNATURE)


def _signer_principal() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "user.email"],
            text=True, stderr=subprocess.DEVNULL).strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def verify_manifest(bundle_dir: str, principal: Optional[str] = None
                    ) -> Tuple[bool, List[str]]:
    """Verify the manifest signature (if present) and that every file's
    on-disk hash matches the manifest. Returns (ok, problems)."""
    import json
    problems: List[str] = []
    manifest_path = os.path.join(bundle_dir, MANIFEST)
    sig_path = os.path.join(bundle_dir, SIGNATURE)
    if not os.path.isfile(manifest_path):
        return False, [f"no {MANIFEST}"]

    # 1. Signature (when present — a bundle may be hash-only Preview).
    if os.path.isfile(sig_path):
        who = principal or _signer_principal()
        if not who:
            problems.append("cannot determine signer principal to verify")
        else:
            try:
                with open(manifest_path, "rb") as mf:
                    subprocess.run(
                        ["ssh-keygen", "-Y", "verify", "-f",
                         _allowed_signers(), "-I", who, "-n", NAMESPACE,
                         "-s", sig_path],
                        stdin=mf, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (subprocess.SubprocessError, OSError):
                problems.append("manifest signature did not verify against "
                                "allowed_signers")
    else:
        problems.append("unsigned (no signature; Preview-level trust only)")

    # 2. File hashes vs manifest — catches tampering even if signed.
    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return False, problems + ["manifest unreadable"]
    for name, expected in manifest.get("artifact_sha256", {}).items():
        actual = _sha256(os.path.join(bundle_dir, name))
        if actual is None:
            problems.append(f"{name}: listed in manifest but missing on disk")
        elif actual != expected:
            problems.append(f"{name}: hash mismatch (tampered or stale)")

    # Signed + all hashes good = fully verified.
    ok = not [p for p in problems if "Preview-level" not in p]
    return ok, problems
