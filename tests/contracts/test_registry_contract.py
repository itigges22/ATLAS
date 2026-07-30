"""Model-registry artifact contract check.

Every registry entry must be installable or carry a documented gated
exception; every downloadable artifact must be hash-pinned; artifact
filenames must be ones the runtime actually consumes.
"""

from atlas.commands import model_registry

# Artifact filenames with a runtime consumer (lens loader / doctor /
# llama entrypoint). A registry entry referencing anything else is
# shipping bytes nothing reads.
CONSUMED_LENS_FILES = {
    "cost_field.pt", "cost_field.safetensors",
    "gx_xgboost.json", "gx_weights.json", "gx_thresholds.json",
    "cx_normalization.json", "model_identity.json",
}
CONSUMED_ASA_FILES = {"ast_edit_steering.gguf"}

LENS_STATUSES = {"supported", "no-artifacts", "unverified"}
ASA_STATUSES = {"supported", "no-artifacts", "unverified"}


def test_entries_have_identity_fields():
    for m in model_registry.REGISTRY:
        assert m.name and m.tier and m.model_file and m.model_display, m
        assert m.lens_status in LENS_STATUSES, (m.name, m.lens_status)
        assert m.asa_status in ASA_STATUSES, (m.name, m.asa_status)


def test_public_downloads_are_hash_pinned():
    """A download URL without requires_hf_token must carry a SHA-256 —
    the installer verifies it, and shipping an unpinned public URL
    reopens the unverified-download path."""
    for m in model_registry.REGISTRY:
        if m.download_url and not m.requires_hf_token:
            assert m.sha256 and len(m.sha256) == 64, (
                f"{m.name}: public download_url without a sha256 pin")


def test_gated_entries_document_why_hash_is_absent():
    """requires_hf_token is the documented exception for a missing
    hash (anonymous HEAD is impossible); anything else lacking both a
    URL and a hash must explain itself in notes."""
    for m in model_registry.REGISTRY:
        if m.sha256 is None and not m.requires_hf_token:
            assert m.download_url is None, (
                f"{m.name}: has a URL but neither sha256 nor the "
                "gated-source exception")
            assert m.notes, f"{m.name}: uninstallable with no explanation"


def test_artifact_url_bases_pin_every_file():
    for m in model_registry.REGISTRY:
        if m.lens_artifact_url_base:
            missing = [f for f in m.lens_artifact_files
                       if f not in m.lens_artifact_sha256]
            assert not missing, (
                f"{m.name}: lens artifacts downloadable but unpinned: "
                f"{missing}")
        if m.asa_artifact_url_base:
            missing = [f for f in m.asa_artifact_files
                       if f not in m.asa_artifact_sha256]
            assert not missing, (
                f"{m.name}: ASA artifacts downloadable but unpinned: "
                f"{missing}")


def test_artifact_files_are_consumed_by_runtime():
    for m in model_registry.REGISTRY:
        stray = set(m.lens_artifact_files) - CONSUMED_LENS_FILES
        assert not stray, (
            f"{m.name}: lens_artifact_files names files the runtime "
            f"never reads: {sorted(stray)}")
        stray = set(m.asa_artifact_files) - CONSUMED_ASA_FILES
        assert not stray, (
            f"{m.name}: asa_artifact_files names files the runtime "
            f"never reads: {sorted(stray)}")
