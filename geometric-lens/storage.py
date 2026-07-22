import os
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


# Project IDs from generate_project_id() match `^proj_[0-9a-f]{16}$`,
# but create_project() / get_metadata() / delete_project() accept the
# project_id as a function arg — meaning a malicious caller (or a bug
# in the HTTP layer) could inject `../../etc/passwd` and escape
# base_path. _SAFE_ID is the gate: anything that doesn't match gets
# refused with ValueError before it can join into a path. Pattern allows
# the generated form plus a permissive alnum/dash/underscore set so
# legitimate non-generated IDs still work.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _safe_project_id(project_id: str) -> str:
    """Validate a project_id before it's used in a path join. Raises
    ValueError on anything that could escape the project base dir
    (slashes, parent-references, control chars, empty / overlong)."""
    if not isinstance(project_id, str) or not _SAFE_ID.match(project_id):
        raise ValueError(
            f"invalid project_id (must match {_SAFE_ID.pattern})")
    return project_id


@dataclass
class ProjectMetadata:
    project_id: str
    project_name: str
    project_hash: str
    files_indexed: int
    chunks_created: int
    loc_indexed: int
    size_bytes: int
    created_at: str
    expires_at: str
    status: str = "synced"


class ProjectStore:
    """Simple file-based project storage."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.environ.get("PROJECT_DATA_DIR", "/data/projects")
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _project_path(self, project_id: str) -> str:
        # Chokepoint: every downstream path lookup flows through here, so
        # one validation here protects the whole class without touching
        # individual call sites.
        return os.path.join(self.base_path, _safe_project_id(project_id))

    def _metadata_path(self, project_id: str) -> str:
        return os.path.join(self._project_path(project_id), "metadata.json")

    def _files_path(self, project_id: str) -> str:
        return os.path.join(self._project_path(project_id), "files.json")

    def generate_project_id(self, project_name: str, api_key: str) -> str:
        """Generate a unique project ID based on name and API key.

        NOT credential storage — we never compare a stored value against
        this hash. The api_key is here purely as a per-user namespacing
        salt so the same project_name across different keys produces
        different IDs. SHA-256 is appropriate for an opaque uniqueness
        token; CodeQL flags this as py/weak-sensitive-data-hashing
        because of the `api_key` in the input but the alert misreads
        the intent.
        """
        content = f"{api_key}:{project_name}"
        return "proj_" + hashlib.sha256(content.encode()).hexdigest()[:16]

    def project_exists(self, project_id: str) -> bool:
        """Check if a project exists."""
        return os.path.exists(self._metadata_path(project_id))

    def get_project_by_hash(self, project_hash: str, api_key: str) -> Optional[str]:
        """Find a project by its content hash."""
        for project_id in os.listdir(self.base_path):
            # base_path can contain non-conforming entries (hidden dirs,
            # leftover state from older schemas) — skip those rather than
            # propagating the validator's ValueError up to the caller.
            try:
                meta = self.get_metadata(project_id)
            except ValueError:
                continue
            if meta and meta.project_hash == project_hash:
                # Verify ownership via naming convention (simplified)
                return project_id
        return None

    def create_project(
        self,
        project_id: str,
        project_name: str,
        project_hash: str,
        files: List[Dict[str, str]],
        chunks_created: int,
        ttl_hours: int = 24
    ) -> ProjectMetadata:
        """Create a new project."""
        project_path = self._project_path(project_id)
        os.makedirs(project_path, exist_ok=True)

        # Calculate stats
        loc = sum(len(f["content"].split("\n")) for f in files)
        size = sum(len(f["content"].encode()) for f in files)

        now = datetime.utcnow()
        metadata = ProjectMetadata(
            project_id=project_id,
            project_name=project_name,
            project_hash=project_hash,
            files_indexed=len(files),
            chunks_created=chunks_created,
            loc_indexed=loc,
            size_bytes=size,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=ttl_hours)).isoformat(),
            status="synced"
        )

        # Save metadata
        with open(self._metadata_path(project_id), "w") as f:
            json.dump(asdict(metadata), f)

        # Save files (for potential re-indexing)
        with open(self._files_path(project_id), "w") as f:
            json.dump(files, f)

        # !r escapes control chars; project_id is also validated by
        # _safe_project_id at the path-join chokepoint, but the safe
        # render here defends against future refactors.
        logger.info(f"Created project {project_id!r}: {len(files)} files, {loc} LOC")
        return metadata

    def get_metadata(self, project_id: str) -> Optional[ProjectMetadata]:
        """Get project metadata."""
        meta_path = self._metadata_path(project_id)
        if not os.path.exists(meta_path):
            return None

        with open(meta_path) as f:
            data = json.load(f)
            return ProjectMetadata(**data)

    def get_files(self, project_id: str) -> Optional[List[Dict[str, str]]]:
        """Get project files."""
        files_path = self._files_path(project_id)
        if not os.path.exists(files_path):
            return None

        with open(files_path) as f:
            return json.load(f)

    def update_metadata(self, project_id: str, **kwargs) -> Optional[ProjectMetadata]:
        """Update project metadata."""
        meta = self.get_metadata(project_id)
        if not meta:
            return None

        data = asdict(meta)
        data.update(kwargs)
        meta = ProjectMetadata(**data)

        with open(self._metadata_path(project_id), "w") as f:
            json.dump(asdict(meta), f)

        return meta

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data."""
        import shutil
        project_path = self._project_path(project_id)
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
            logger.info(f"Deleted project {project_id!r}")
            return True
        return False

    def list_projects(self, api_key: Optional[str] = None) -> List[ProjectMetadata]:
        """List all projects."""
        projects = []
        for project_id in os.listdir(self.base_path):
            try:
                meta = self.get_metadata(project_id)
            except ValueError:
                # Non-conforming dir entry — skip silently rather than
                # break listing on a single stray file. Same guard as
                # get_project_by_hash.
                continue
            if meta:
                projects.append(meta)
        return projects

    def cleanup_expired(self) -> int:
        """Delete expired projects."""
        now = datetime.utcnow()
        deleted = 0

        for project_id in os.listdir(self.base_path):
            meta = self.get_metadata(project_id)
            if meta:
                expires = datetime.fromisoformat(meta.expires_at)
                if now > expires:
                    self.delete_project(project_id)
                    deleted += 1

        if deleted:
            logger.info(f"Cleaned up {deleted} expired projects")
        return deleted


# Global store instance
project_store = ProjectStore()
