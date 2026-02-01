import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


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

    def __init__(self, base_path: str = "/data/projects"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _project_path(self, project_id: str) -> str:
        return os.path.join(self.base_path, project_id)

    def _metadata_path(self, project_id: str) -> str:
        return os.path.join(self._project_path(project_id), "metadata.json")

    def _files_path(self, project_id: str) -> str:
        return os.path.join(self._project_path(project_id), "files.json")

    def generate_project_id(self, project_name: str, api_key: str) -> str:
        """Generate a unique project ID based on name and API key."""
        content = f"{api_key}:{project_name}"
        return "proj_" + hashlib.sha256(content.encode()).hexdigest()[:16]

    def project_exists(self, project_id: str) -> bool:
        """Check if a project exists."""
        return os.path.exists(self._metadata_path(project_id))

    def get_project_by_hash(self, project_hash: str, api_key: str) -> Optional[str]:
        """Find a project by its content hash."""
        for project_id in os.listdir(self.base_path):
            meta = self.get_metadata(project_id)
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

        logger.info(f"Created project {project_id}: {len(files)} files, {loc} LOC")
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
            logger.info(f"Deleted project {project_id}")
            return True
        return False

    def list_projects(self, api_key: Optional[str] = None) -> List[ProjectMetadata]:
        """List all projects."""
        projects = []
        for project_id in os.listdir(self.base_path):
            meta = self.get_metadata(project_id)
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
