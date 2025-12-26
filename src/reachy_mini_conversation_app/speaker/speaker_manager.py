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

    Features:
    - Real-time speaker identification (6s embedding window)
    - Background diarization for enrollment (triggered on unknown speaker)
    - Online adaptation of voiceprints with safeguards
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        voiceprint_store: Optional[VoiceprintStore] = None,
        confidence_threshold: Optional[float] = None,
        on_speaker_change: Optional[Callable[[SpeakerResult], None]] = None,
        audio_buffer: Optional[Any] = None,  # AudioBuffer for diarization
        mentioned_names: Optional[Dict[str, float]] = None,  # name -> timestamp for enrollment
    ):
        """Initialize the speaker manager.

        Args:
            voiceprint_store: Store for voiceprints. Created if not provided.
            confidence_threshold: Minimum similarity for identification.
            on_speaker_change: Callback when speaker changes.
            audio_buffer: AudioBuffer instance for background diarization.
            mentioned_names: Dict mapping names to timestamps (for diarization enrollment).
        """
        self.store = voiceprint_store or VoiceprintStore()
        self.threshold = confidence_threshold or config.SPEAKER_CONFIDENCE_THRESHOLD
        self.on_speaker_change = on_speaker_change
        self.audio_buffer = audio_buffer
        self.mentioned_names = mentioned_names  # Reference to shared dict

        self._model: Any = None
        self._inference: Any = None
        self._audio_buffer_internal: list[np.ndarray] = []
        self._buffer_samples = 0

        self._current_speaker: Optional[str] = None
        self._current_confidence: float = 0.0

        self._initialized = False

        # Background diarization for enrollment
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

    def initialize(self) -> bool:
        """Initialize pyannote model and background diarizer.

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
        """Callback when background diarization completes.

        Enrolls speakers who:
        1. Are matched to a name (via timing or voiceprint)
        2. Have sufficient speech duration (>= MIN_ENROLLMENT_SECONDS)
        3. Are not already enrolled
        """
        if not result.get("success"):
            logger.warning(f"Background diarization failed: {result.get('error')}")
            return

        segments = result.get("segments", [])
        if not segments:
            return

        unique_speakers = set(s[2] for s in segments)
        logger.info(f"Diarization found {len(unique_speakers)} speakers in {len(segments)} segments")

        # Get audio for embedding extraction
        if self.audio_buffer is None:
            return
        audio = self.audio_buffer.get_audio()
        if audio is None:
            return

        # Match diarization speakers to names/voiceprints
        from reachy_mini_conversation_app.speaker.diarization import (
            match_speakers_to_names,
            extract_speaker_segments,
        )

        speaker_to_name = match_speakers_to_names(
            segments=segments,
            mentioned_names=self.mentioned_names or {},
            audio=audio,
            sample_rate=self.SAMPLE_RATE,
            store=self.store,
        )

        if not speaker_to_name:
            logger.debug("No speakers matched to names in diarization results")
            return

        # Try to enroll matched speakers
        for speaker_label, name in speaker_to_name.items():
            # Skip already enrolled
            if self.store.get(name) is not None:
                logger.debug(f"Speaker '{name}' already enrolled, skipping")
                continue

            # Check minimum speech duration
            total_speech = sum(
                end - start for start, end, label in segments if label == speaker_label
            )
            if total_speech < config.SPEAKER_MIN_ENROLLMENT_SECONDS:
                logger.info(
                    f"Speaker '{name}' has {total_speech:.1f}s speech, "
                    f"need {config.SPEAKER_MIN_ENROLLMENT_SECONDS}s for enrollment"
                )
                continue

            # Extract clean segments and enroll
            speaker_segments = extract_speaker_segments(
                audio, segments, speaker_label, self.SAMPLE_RATE,
                min_segment_duration=2.0, max_segments=5
            )

            if self._enroll_from_diarization(name, speaker_segments):
                logger.info(f"Enrolled '{name}' from diarization ({total_speech:.1f}s speech)")

    def _enroll_from_diarization(self, name: str, segments: list) -> bool:
        """Enroll a speaker from clean diarization segments.

        Args:
            name: Speaker name to enroll
            segments: List of audio segments (numpy arrays) for the speaker

        Returns:
            True if enrollment succeeded, False otherwise
        """
        if len(segments) < 2:
            logger.warning(f"Not enough segments for '{name}' (need at least 2)")
            return False

        try:
            # Extract embedding from each clean segment
            embeddings = []
            for seg_audio in segments:
                emb = self._get_embedding(seg_audio)
                embeddings.append(emb)

            # Average for robustness
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            # Save voiceprint
            self.store.save(name, avg_embedding)
            return True

        except Exception as e:
            logger.error(f"Failed to enroll '{name}' from diarization: {e}")
            return False

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
        target_samples = int(config.SPEAKER_EMBEDDING_WINDOW_SECONDS * self.SAMPLE_RATE)

        if self._buffer_samples < target_samples:
            return None

        # Concatenate and process
        audio = np.concatenate(self._audio_buffer_internal)
        self._audio_buffer_internal = []
        self._buffer_samples = 0

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

        Compares embedding to enrolled speakers. If unknown, triggers
        background diarization to potentially enroll the speaker.
        """
        try:
            embedding = self._get_embedding(audio)
            result = self._compare_to_enrolled(embedding)

            # Trigger diarization on unknown speaker (for potential enrollment)
            if result.is_unknown and self.background_diarizer is not None and self.audio_buffer is not None:
                triggered = self.background_diarizer.trigger(self.audio_buffer)
                if triggered:
                    logger.info("Background diarization triggered for unknown speaker")

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
        - Adaptation counts (per-session limits)
        """
        # Reset adaptation counts for new session
        self._adaptation_counts.clear()
        # Note: _last_adaptation_time is kept to respect cooldowns across sessions

        logger.info("Speaker manager session reset")

    def stop(self) -> None:
        """Stop background processes and clean up."""
        if self.background_diarizer is not None:
            self.background_diarizer.stop()
