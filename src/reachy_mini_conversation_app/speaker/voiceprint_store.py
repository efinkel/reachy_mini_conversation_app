"""Voiceprint storage for speaker embeddings."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


class VoiceprintStore:
    """Manages storage and retrieval of speaker voiceprints (embeddings)."""

    def __init__(self, voiceprint_dir: Optional[str] = None):
        """Initialize the voiceprint store.

        Args:
            voiceprint_dir: Directory for voiceprints. Defaults to config.VOICEPRINT_DIR.
        """
        self.voiceprint_dir = Path(voiceprint_dir or config.VOICEPRINT_DIR)
        self.manifest_path = self.voiceprint_dir / "manifest.json"
        self._speakers: Dict[str, np.ndarray] = {}
        self._load_all()

    def _ensure_dir(self) -> None:
        """Create voiceprint directory if it doesn't exist."""
        self.voiceprint_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> Dict[str, str]:
        """Load the manifest mapping speaker names to files."""
        if not self.manifest_path.exists():
            return {}
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load manifest: {e}")
            return {}

    def _save_manifest(self, manifest: Dict[str, str]) -> None:
        """Save the manifest to disk."""
        self._ensure_dir()
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_all(self) -> None:
        """Load all voiceprints from disk."""
        manifest = self._load_manifest()
        for name, filename in manifest.items():
            filepath = self.voiceprint_dir / filename
            if filepath.exists():
                try:
                    self._speakers[name] = np.load(filepath)
                    logger.debug(f"Loaded voiceprint for '{name}'")
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to load voiceprint for '{name}': {e}")
        if self._speakers:
            logger.info(f"Loaded {len(self._speakers)} voiceprint(s): {list(self._speakers.keys())}")

    def save(self, name: str, embedding: np.ndarray) -> None:
        """Save a voiceprint for a speaker.

        Args:
            name: Speaker name (will be lowercased)
            embedding: Speaker embedding vector
        """
        name = name.lower().strip()
        if not name:
            raise ValueError("Speaker name cannot be empty")

        self._ensure_dir()
        filename = f"{name}.npy"
        filepath = self.voiceprint_dir / filename

        np.save(filepath, embedding)
        self._speakers[name] = embedding

        manifest = self._load_manifest()
        manifest[name] = filename
        self._save_manifest(manifest)

        logger.info(f"Saved voiceprint for '{name}'")

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get a voiceprint by speaker name.

        Args:
            name: Speaker name

        Returns:
            Embedding vector or None if not found
        """
        return self._speakers.get(name.lower().strip())

    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all enrolled voiceprints.

        Returns:
            Dict mapping speaker names to embeddings
        """
        return self._speakers.copy()

    def remove(self, name: str) -> bool:
        """Remove a voiceprint.

        Args:
            name: Speaker name

        Returns:
            True if removed, False if not found
        """
        name = name.lower().strip()
        if name not in self._speakers:
            return False

        del self._speakers[name]

        manifest = self._load_manifest()
        if name in manifest:
            filepath = self.voiceprint_dir / manifest[name]
            try:
                filepath.unlink(missing_ok=True)
            except OSError:
                pass
            del manifest[name]
            self._save_manifest(manifest)

        logger.info(f"Removed voiceprint for '{name}'")
        return True

    def list_speakers(self) -> list[str]:
        """List all enrolled speaker names."""
        return list(self._speakers.keys())
