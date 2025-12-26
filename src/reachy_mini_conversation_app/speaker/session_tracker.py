"""Session-based speaker tracking for within-conversation identification."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


@dataclass
class SessionSpeaker:
    """Represents a speaker detected within a single session."""

    session_id: str  # e.g., "session_speaker_0"
    embeddings: List[np.ndarray] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    matched_name: Optional[str] = None
    is_enrolled: bool = False  # Already in voiceprint store
    known_speaker_id: Optional[str] = None  # If matched to enrolled speaker

    def add_embedding(self, embedding: np.ndarray, timestamp: float) -> None:
        """Add an embedding observation for this speaker."""
        self.embeddings.append(embedding)
        if self.first_seen == 0.0:
            self.first_seen = timestamp
        self.last_seen = timestamp

    def get_averaged_embedding(self) -> Optional[np.ndarray]:
        """Get the averaged (normalized) embedding for this speaker."""
        if not self.embeddings:
            return None
        avg = np.mean(self.embeddings, axis=0)
        return avg / np.linalg.norm(avg)

    @property
    def total_speech_time(self) -> float:
        """Estimate total speech time based on embedding count (6s per embedding)."""
        return len(self.embeddings) * config.SPEAKER_EMBEDDING_WINDOW_SECONDS


class SessionSpeakerTracker:
    """Tracks speakers within a single conversation session.

    Uses real-time embeddings to identify and track speakers during a session.
    When a new speaker is detected, it signals that background diarization
    should be triggered.
    """

    def __init__(
        self,
        similarity_threshold: float = None,
    ):
        """Initialize the session tracker.

        Args:
            similarity_threshold: Cosine similarity threshold for speaker matching.
                                  Higher = stricter matching. Defaults to config value.
        """
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else config.SPEAKER_SESSION_SIMILARITY_THRESHOLD
        )

        self.speakers: Dict[str, SessionSpeaker] = {}
        self.current_speaker_id: Optional[str] = None
        self._next_speaker_idx: int = 0

        # Track known (enrolled) speakers seen this session
        self._known_speakers_seen: Dict[str, str] = {}  # known_id -> session_id

    def process_embedding(
        self,
        embedding: np.ndarray,
        timestamp: float,
        known_speaker_id: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Process a new embedding and assign to a speaker.

        Args:
            embedding: Speaker embedding vector (512-dim from pyannote)
            timestamp: When this embedding was extracted
            known_speaker_id: If not None, this embedding was identified as
                              an enrolled speaker with this ID

        Returns:
            Tuple of (session_speaker_id, is_new_to_session)
        """
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # If this is a known speaker, map to existing session speaker or create one
        if known_speaker_id is not None:
            if known_speaker_id in self._known_speakers_seen:
                session_id = self._known_speakers_seen[known_speaker_id]
                speaker = self.speakers[session_id]
                speaker.add_embedding(embedding, timestamp)
                self.current_speaker_id = session_id
                return session_id, False
            else:
                # First time seeing this known speaker in this session
                # Check if we should merge with the current/recent unknown speaker
                merged_session_id = self._try_merge_current_to_known(known_speaker_id)
                if merged_session_id:
                    # Merged with existing session speaker
                    speaker = self.speakers[merged_session_id]
                    speaker.add_embedding(embedding, timestamp)
                    self.current_speaker_id = merged_session_id
                    logger.info(f"Known speaker '{known_speaker_id}' merged with {merged_session_id}")
                    return merged_session_id, False
                else:
                    # Create new session speaker for this known speaker
                    session_id = self._create_speaker(
                        embedding, timestamp,
                        is_enrolled=True,
                        known_speaker_id=known_speaker_id
                    )
                    self._known_speakers_seen[known_speaker_id] = session_id
                    logger.info(f"Known speaker '{known_speaker_id}' entered session as {session_id}")
                    return session_id, True

        # Unknown speaker - compare to existing session speakers
        best_match: Optional[str] = None
        best_similarity: float = 0.0

        for session_id, speaker in self.speakers.items():
            if speaker.is_enrolled:
                # Skip enrolled speakers - they should be matched via known_speaker_id
                continue

            speaker_embedding = speaker.get_averaged_embedding()
            if speaker_embedding is None:
                continue

            similarity = 1 - cosine(embedding, speaker_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = session_id

        # Check if match is good enough
        if best_match is not None and best_similarity >= self.similarity_threshold:
            speaker = self.speakers[best_match]
            speaker.add_embedding(embedding, timestamp)
            self.current_speaker_id = best_match
            return best_match, False

        # New speaker in this session
        session_id = self._create_speaker(embedding, timestamp)
        logger.info(f"New session speaker detected: {session_id} (similarity={best_similarity:.2f})")
        return session_id, True

    def _create_speaker(
        self,
        embedding: np.ndarray,
        timestamp: float,
        is_enrolled: bool = False,
        known_speaker_id: Optional[str] = None,
    ) -> str:
        """Create a new session speaker."""
        session_id = f"session_speaker_{self._next_speaker_idx}"
        self._next_speaker_idx += 1

        speaker = SessionSpeaker(
            session_id=session_id,
            is_enrolled=is_enrolled,
            known_speaker_id=known_speaker_id,
        )
        speaker.add_embedding(embedding, timestamp)
        self.speakers[session_id] = speaker
        self.current_speaker_id = session_id

        return session_id

    def _try_merge_current_to_known(self, known_speaker_id: str) -> Optional[str]:
        """Try to merge the current unknown session speaker with a known speaker.

        When we first identify a known speaker, check if the current/recent
        session speaker is likely the same person. If so, merge them.

        Args:
            known_speaker_id: The enrolled speaker ID that was just identified

        Returns:
            Session ID if merge happened, None otherwise
        """
        # If there's no current speaker or it's already enrolled, no merge needed
        if self.current_speaker_id is None:
            return None

        current = self.speakers.get(self.current_speaker_id)
        if current is None or current.is_enrolled:
            return None

        # Merge the current unknown speaker into the known speaker
        current.is_enrolled = True
        current.known_speaker_id = known_speaker_id
        self._known_speakers_seen[known_speaker_id] = self.current_speaker_id
        logger.debug(f"Merged {self.current_speaker_id} into known speaker '{known_speaker_id}'")
        return self.current_speaker_id

    def assign_name_to_recent(
        self,
        name: str,
        timestamp: float,
        lookback_seconds: float = 10.0,
    ) -> bool:
        """Assign a name to the speaker who was active near a timestamp.

        Called when someone says "I'm [name]" or similar.

        Args:
            name: The name to assign
            timestamp: When the name was mentioned
            lookback_seconds: How far back to look for active speaker

        Returns:
            True if name was assigned, False otherwise
        """
        name_lower = name.lower()

        # Find speaker active at or shortly before this timestamp
        best_speaker: Optional[SessionSpeaker] = None
        best_time_diff: float = float("inf")

        for speaker in self.speakers.values():
            # Skip already-enrolled speakers
            if speaker.is_enrolled:
                continue
            # Skip if already has a different name
            if speaker.matched_name and speaker.matched_name != name_lower:
                continue

            # Check if this speaker was active near the timestamp
            time_diff = timestamp - speaker.last_seen
            if 0 <= time_diff <= lookback_seconds and time_diff < best_time_diff:
                best_time_diff = time_diff
                best_speaker = speaker

        if best_speaker is not None:
            best_speaker.matched_name = name_lower
            logger.info(f"Assigned name '{name}' to {best_speaker.session_id}")
            return True

        logger.debug(f"Could not assign name '{name}' - no recent speaker found")
        return False

    def update_from_diarization(
        self,
        segments: List[Tuple[float, float, str]],
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> None:
        """Update session speakers based on diarization results.

        This can refine speaker boundaries and potentially merge/split speakers.

        Args:
            segments: Diarization segments (start, end, speaker_label)
            audio: The audio that was diarized
            sample_rate: Audio sample rate
        """
        # For now, just log the update
        # More sophisticated merging/refinement could be added later
        unique_speakers = set(s[2] for s in segments)
        logger.info(
            f"Diarization found {len(unique_speakers)} speakers in {len(segments)} segments"
        )

    def get_unenrolled_with_names(self) -> List[SessionSpeaker]:
        """Get speakers who have names but aren't enrolled yet.

        These are candidates for enrollment at session end.

        Returns:
            List of SessionSpeaker objects ready for enrollment
        """
        return [
            speaker for speaker in self.speakers.values()
            if speaker.matched_name is not None
            and not speaker.is_enrolled
            and len(speaker.embeddings) >= config.SPEAKER_MIN_EMBEDDINGS_FOR_ENROLLMENT
        ]

    def get_all_session_speakers(self) -> List[SessionSpeaker]:
        """Get all speakers tracked in this session."""
        return list(self.speakers.values())

    def get_speaker_count(self) -> int:
        """Get the number of unique speakers in this session."""
        return len(self.speakers)

    def get_current_speaker(self) -> Optional[SessionSpeaker]:
        """Get the most recently active speaker."""
        if self.current_speaker_id is None:
            return None
        return self.speakers.get(self.current_speaker_id)

    def reset(self) -> None:
        """Reset the tracker for a new session."""
        self.speakers.clear()
        self.current_speaker_id = None
        self._next_speaker_idx = 0
        self._known_speakers_seen.clear()
