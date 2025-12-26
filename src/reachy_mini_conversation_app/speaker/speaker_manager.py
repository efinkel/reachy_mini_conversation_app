"""Speaker identification manager using pyannote.audio."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.spatial.distance import cosine

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.speaker.voiceprint_store import VoiceprintStore


logger = logging.getLogger(__name__)


@dataclass
class SpeakerResult:
    """Result of speaker identification."""

    speaker: Optional[str]
    confidence: float
    is_unknown: bool

    @property
    def is_identified(self) -> bool:
        """True if a known speaker was identified."""
        return self.speaker is not None and not self.is_unknown


class SpeakerManager:
    """Manages speaker identification using pyannote embeddings.

    Now includes:
    - Session tracking for within-conversation speaker management
    - Background diarization triggered on new speaker detection
    - Online adaptation of voiceprints with safeguards
    """

    SAMPLE_RATE = 16000
    AUDIO_DURATION_ENROLL = 5.0  # seconds needed for enrollment (more for quality)

    def __init__(
        self,
        voiceprint_store: Optional[VoiceprintStore] = None,
        confidence_threshold: Optional[float] = None,
        on_speaker_change: Optional[Callable[[SpeakerResult], None]] = None,
        audio_buffer: Optional[Any] = None,  # AudioBuffer for diarization
    ):
        """Initialize the speaker manager.

        Args:
            voiceprint_store: Store for voiceprints. Created if not provided.
            confidence_threshold: Minimum similarity for identification.
            on_speaker_change: Callback when speaker changes.
            audio_buffer: AudioBuffer instance for background diarization.
        """
        self.store = voiceprint_store or VoiceprintStore()
        self.threshold = confidence_threshold or config.SPEAKER_CONFIDENCE_THRESHOLD
        self.on_speaker_change = on_speaker_change
        self.audio_buffer = audio_buffer

        self._model: Any = None
        self._inference: Any = None
        self._audio_buffer_internal: list[np.ndarray] = []
        self._buffer_samples = 0

        self._current_speaker: Optional[str] = None
        self._current_confidence: float = 0.0

        self._enrolling: bool = False
        self._enroll_name: Optional[str] = None
        self._enroll_callback: Optional[Callable[[bool, str], None]] = None

        self._initialized = False

        # Session tracking and background diarization
        self.session_tracker: Optional[Any] = None  # SessionSpeakerTracker
        self.background_diarizer: Optional[Any] = None  # BackgroundDiarizer

        # Online adaptation tracking (per speaker)
        self._last_adaptation_time: Dict[str, float] = {}
        self._adaptation_counts: Dict[str, int] = {}

    @property
    def current_speaker(self) -> Optional[str]:
        """Currently identified speaker, or None."""
        return self._current_speaker

    @property
    def current_confidence(self) -> float:
        """Confidence of current speaker identification."""
        return self._current_confidence

    @property
    def is_enrolling(self) -> bool:
        """True if currently enrolling a new speaker."""
        return self._enrolling

    def initialize(self) -> bool:
        """Initialize pyannote model, session tracker, and background diarizer.

        Returns:
            True if successful, False if pyannote not available.
        """
        if self._initialized:
            return True

        if not config.HF_TOKEN:
            logger.warning("HF_TOKEN not set, speaker identification disabled")
            return False

        try:
            from pyannote.audio import Inference, Model

            logger.info("Loading pyannote embedding model...")
            self._model = Model.from_pretrained("pyannote/embedding", token=config.HF_TOKEN)
            self._inference = Inference(self._model, window="whole")

            # Initialize session tracker
            from reachy_mini_conversation_app.speaker.session_tracker import SessionSpeakerTracker
            self.session_tracker = SessionSpeakerTracker()
            logger.info("Session speaker tracker initialized")

            # Initialize background diarizer if we have an audio buffer
            if self.audio_buffer is not None:
                from reachy_mini_conversation_app.speaker.background_diarizer import BackgroundDiarizer
                self.background_diarizer = BackgroundDiarizer(
                    window_seconds=config.SPEAKER_DIARIZATION_WINDOW_SECONDS,
                    cooldown_seconds=config.SPEAKER_DIARIZATION_COOLDOWN_SECONDS,
                    on_results=self._on_diarization_results,
                )
                logger.info("Background diarizer initialized")

            self._initialized = True
            logger.info("Speaker identification initialized")
            return True
        except ImportError:
            logger.warning("pyannote.audio not installed, speaker identification disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pyannote: {e}")
            return False

    def _on_diarization_results(self, result: Dict[str, Any]) -> None:
        """Callback when background diarization completes."""
        if not result.get("success"):
            logger.warning(f"Background diarization failed: {result.get('error')}")
            return

        segments = result.get("segments", [])
        if segments and self.session_tracker is not None:
            # Update session tracker with diarization results
            # For now, just log - full integration would update speaker boundaries
            unique_speakers = set(s[2] for s in segments)
            logger.info(f"Diarization found {len(unique_speakers)} speakers")

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio samples (mono, 16kHz, float32)

        Returns:
            512-dimensional embedding vector
        """
        import torch

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add channel dimension

        waveform = torch.from_numpy(audio.astype(np.float32))
        embedding = self._inference({"waveform": waveform, "sample_rate": self.SAMPLE_RATE})
        return np.array(embedding)

    def _compare_to_enrolled(self, embedding: np.ndarray) -> SpeakerResult:
        """Compare embedding to enrolled speakers.

        Args:
            embedding: Query embedding

        Returns:
            SpeakerResult with best match
        """
        enrolled = self.store.get_all()
        if not enrolled:
            return SpeakerResult(speaker=None, confidence=0.0, is_unknown=True)

        best_speaker = None
        best_similarity = 0.0

        for name, stored_embedding in enrolled.items():
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = name

        if best_similarity >= self.threshold:
            return SpeakerResult(speaker=best_speaker, confidence=best_similarity, is_unknown=False)
        else:
            return SpeakerResult(speaker=None, confidence=best_similarity, is_unknown=True)

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[SpeakerResult]:
        """Process an audio chunk for speaker identification.

        Call this with audio frames as they arrive. Returns a result
        when enough audio has been accumulated for identification.

        Args:
            audio_chunk: Audio samples (mono, 16kHz, float32)

        Returns:
            SpeakerResult when identification complete, None otherwise
        """
        if not self._initialized:
            return None

        self._audio_buffer_internal.append(audio_chunk)
        self._buffer_samples += len(audio_chunk)

        # Use config-based duration for identification (6 seconds default)
        target_duration = self.AUDIO_DURATION_ENROLL if self._enrolling else config.SPEAKER_EMBEDDING_WINDOW_SECONDS
        target_samples = int(target_duration * self.SAMPLE_RATE)

        if self._buffer_samples < target_samples:
            return None

        # Concatenate and process
        audio = np.concatenate(self._audio_buffer_internal)
        self._audio_buffer_internal = []
        self._buffer_samples = 0

        if self._enrolling:
            return self._complete_enrollment(audio)
        else:
            return self._identify_speaker(audio)

    def check_background_results(self) -> Optional[Dict[str, Any]]:
        """Check for completed background diarization results.

        Should be called periodically from the main loop.

        Returns:
            Diarization result dict if available, None otherwise.
        """
        if self.background_diarizer is None:
            return None
        return self.background_diarizer.check_results()

    def _identify_speaker(self, audio: np.ndarray) -> SpeakerResult:
        """Identify speaker from audio.

        Now includes:
        - Session tracking for within-conversation speaker management
        - Online adaptation of known speaker voiceprints (with safeguards)
        - Background diarization trigger on new speaker detection
        """
        try:
            embedding = self._get_embedding(audio)
            result = self._compare_to_enrolled(embedding)
            timestamp = time.time()

            # Track in session and potentially trigger background diarization
            if self.session_tracker is not None:
                known_speaker_id = result.speaker if result.is_identified else None
                session_id, is_new_to_session = self.session_tracker.process_embedding(
                    embedding, timestamp, known_speaker_id
                )

                # If new speaker detected, trigger background diarization
                if is_new_to_session and self.background_diarizer is not None and self.audio_buffer is not None:
                    triggered = self.background_diarizer.trigger(self.audio_buffer)
                    if triggered:
                        logger.info(f"Background diarization triggered for new speaker: {session_id}")

            # Online adaptation for known speakers (with safeguards)
            if result.is_identified:
                self._maybe_adapt_voiceprint(result.speaker, embedding, result.confidence)

            # Handle speaker change callbacks
            if result.speaker != self._current_speaker:
                self._current_speaker = result.speaker
                self._current_confidence = result.confidence
                if self.on_speaker_change:
                    self.on_speaker_change(result)
                if result.is_identified:
                    logger.info(f"Speaker identified: {result.speaker} ({result.confidence:.2f})")
                else:
                    logger.debug(f"Unknown speaker (best match: {result.confidence:.2f})")
            else:
                self._current_confidence = result.confidence

            return result
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return SpeakerResult(speaker=None, confidence=0.0, is_unknown=True)

    def _maybe_adapt_voiceprint(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        confidence: float,
    ) -> bool:
        """Adapt voiceprint only if all safeguards pass.

        Safeguards:
        1. High confidence threshold (stricter than identification)
        2. Rate limiting (max 1 adaptation per speaker per minute)
        3. Session cap (max N adaptations per speaker per session)

        Args:
            speaker_id: Name of the enrolled speaker
            embedding: New embedding to blend in
            confidence: Confidence of the identification

        Returns:
            True if adaptation was performed, False otherwise
        """
        # Safeguard 1: High confidence threshold
        if confidence < config.SPEAKER_ADAPTATION_MIN_CONFIDENCE:
            return False

        # Safeguard 2: Rate limiting (1 per minute per speaker)
        now = time.time()
        last_adapt = self._last_adaptation_time.get(speaker_id, 0)
        if (now - last_adapt) < config.SPEAKER_ADAPTATION_COOLDOWN_SECONDS:
            return False

        # Safeguard 3: Session cap
        adapt_count = self._adaptation_counts.get(speaker_id, 0)
        if adapt_count >= config.SPEAKER_ADAPTATION_MAX_PER_SESSION:
            return False

        # All safeguards passed - adapt voiceprint
        current_voiceprint = self.store.get(speaker_id)
        if current_voiceprint is None:
            return False

        # Exponential moving average blend
        alpha = config.SPEAKER_ADAPTATION_ALPHA
        new_voiceprint = (1 - alpha) * current_voiceprint + alpha * embedding
        new_voiceprint = new_voiceprint / np.linalg.norm(new_voiceprint)

        self.store.save(speaker_id, new_voiceprint)
        self._last_adaptation_time[speaker_id] = now
        self._adaptation_counts[speaker_id] = adapt_count + 1

        logger.debug(f"Adapted voiceprint for '{speaker_id}' (confidence={confidence:.2f}, count={adapt_count + 1})")
        return True

    def _complete_enrollment(self, audio: np.ndarray) -> SpeakerResult:
        """Complete speaker enrollment with collected audio."""
        name = self._enroll_name
        callback = self._enroll_callback
        self._enrolling = False
        self._enroll_name = None
        self._enroll_callback = None

        try:
            embedding = self._get_embedding(audio)
            self.store.save(name, embedding)
            logger.info(f"Enrolled speaker: {name}")

            if callback:
                callback(True, f"Voice enrolled for {name}")

            return SpeakerResult(speaker=name, confidence=1.0, is_unknown=False)
        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            if callback:
                callback(False, str(e))
            return SpeakerResult(speaker=None, confidence=0.0, is_unknown=True)

    def start_enrollment(
        self,
        name: str,
        callback: Optional[Callable[[bool, str], None]] = None,
    ) -> bool:
        """Start enrolling a new speaker.

        Args:
            name: Name for the new speaker
            callback: Called with (success, message) when enrollment completes

        Returns:
            True if enrollment started, False if not initialized or already enrolling
        """
        if not self._initialized:
            if callback:
                callback(False, "Speaker identification not initialized")
            return False

        if self._enrolling:
            if callback:
                callback(False, "Already enrolling a speaker")
            return False

        self._enrolling = True
        self._enroll_name = name.lower().strip()
        self._enroll_callback = callback
        self._audio_buffer_internal = []
        self._buffer_samples = 0

        logger.info(f"Started enrollment for: {self._enroll_name}")
        return True

    def cancel_enrollment(self) -> None:
        """Cancel ongoing enrollment."""
        if self._enrolling:
            if self._enroll_callback:
                self._enroll_callback(False, "Enrollment cancelled")
            self._enrolling = False
            self._enroll_name = None
            self._enroll_callback = None
            self._audio_buffer_internal = []
            self._buffer_samples = 0
            logger.info("Enrollment cancelled")

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self._audio_buffer_internal = []
        self._buffer_samples = 0

    def list_enrolled_speakers(self) -> list[str]:
        """List all enrolled speaker names."""
        return self.store.list_speakers()

    def remove_speaker(self, name: str) -> bool:
        """Remove an enrolled speaker."""
        return self.store.remove(name)

    def reset_session(self) -> None:
        """Reset session-specific state for a new conversation.

        Call this at the start of each new conversation to clear:
        - Session speaker tracker
        - Adaptation counts (per-session limits)
        - Background diarization state
        """
        if self.session_tracker is not None:
            self.session_tracker.reset()

        # Reset adaptation counts for new session
        self._adaptation_counts.clear()
        # Note: _last_adaptation_time is kept to respect cooldowns across sessions

        logger.info("Speaker manager session reset")

    def get_unenrolled_speakers_with_names(self) -> list:
        """Get session speakers who have names but aren't enrolled.

        Returns:
            List of SessionSpeaker objects ready for enrollment.
        """
        if self.session_tracker is None:
            return []
        return self.session_tracker.get_unenrolled_with_names()

    def assign_name_to_speaker(self, name: str, timestamp: float) -> bool:
        """Assign a name to the speaker active near a timestamp.

        Call this when the user says "I'm [name]" or similar.

        Args:
            name: The name to assign
            timestamp: When the name was mentioned

        Returns:
            True if name was assigned, False otherwise
        """
        if self.session_tracker is None:
            return False
        return self.session_tracker.assign_name_to_recent(name, timestamp)

    def enroll_session_speaker(self, session_speaker) -> bool:
        """Enroll a session speaker using their accumulated embeddings.

        Args:
            session_speaker: SessionSpeaker object with matched_name set

        Returns:
            True if enrollment succeeded, False otherwise
        """
        if session_speaker.matched_name is None:
            logger.warning("Cannot enroll speaker without a name")
            return False

        if session_speaker.is_enrolled:
            logger.debug(f"Speaker '{session_speaker.matched_name}' already enrolled")
            return True

        avg_embedding = session_speaker.get_averaged_embedding()
        if avg_embedding is None:
            logger.warning(f"No embeddings for speaker '{session_speaker.matched_name}'")
            return False

        # Check if we should update existing or create new
        existing = self.store.get(session_speaker.matched_name)
        if existing is not None:
            # Blend with existing (weighted average favoring new)
            blended = 0.3 * existing + 0.7 * avg_embedding
            blended = blended / np.linalg.norm(blended)
            self.store.save(session_speaker.matched_name, blended)
            logger.info(f"Updated voiceprint for '{session_speaker.matched_name}' from session")
        else:
            self.store.save(session_speaker.matched_name, avg_embedding)
            logger.info(f"Created voiceprint for '{session_speaker.matched_name}' from session")

        session_speaker.is_enrolled = True
        return True

    def stop(self) -> None:
        """Stop background processes and clean up."""
        if self.background_diarizer is not None:
            self.background_diarizer.stop()
