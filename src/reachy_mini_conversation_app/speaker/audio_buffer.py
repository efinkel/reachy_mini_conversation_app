"""Rolling audio buffer for speaker diarization."""

import logging
from collections import deque
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class AudioBuffer:
    """Rolling buffer that accumulates audio up to a maximum duration.

    Used to collect audio during a conversation for post-session diarization.
    When the buffer exceeds max_duration, oldest chunks are discarded.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 900.0,  # 15 minutes default
        min_duration: float = 60.0,   # Minimum for enrollment
    ):
        """Initialize the audio buffer.

        Args:
            sample_rate: Audio sample rate in Hz
            max_duration: Maximum buffer duration in seconds (rolling window)
            min_duration: Minimum duration required for enrollment
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_samples = int(max_duration * sample_rate)
        self.min_samples = int(min_duration * sample_rate)

        self._chunks: deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._start_time: Optional[float] = None

    def append(self, audio_chunk: np.ndarray) -> None:
        """Append an audio chunk to the buffer.

        Args:
            audio_chunk: Audio samples (mono, float32)
        """
        if audio_chunk.ndim != 1:
            audio_chunk = audio_chunk.flatten()

        self._chunks.append(audio_chunk)
        self._total_samples += len(audio_chunk)

        # Trim old chunks if we exceed max duration
        while self._total_samples > self.max_samples and self._chunks:
            removed = self._chunks.popleft()
            self._total_samples -= len(removed)

    def get_audio(self) -> Optional[np.ndarray]:
        """Get all buffered audio as a single array.

        Returns:
            Concatenated audio array, or None if buffer is empty
        """
        if not self._chunks:
            return None
        return np.concatenate(list(self._chunks))

    def get_recent(self, seconds: float) -> Optional[np.ndarray]:
        """Get the most recent N seconds of audio.

        Args:
            seconds: How many seconds of audio to retrieve

        Returns:
            Audio array with at most `seconds` worth of samples,
            or None if buffer is empty
        """
        if not self._chunks:
            return None

        target_samples = int(seconds * self.sample_rate)
        all_audio = np.concatenate(list(self._chunks))

        if len(all_audio) <= target_samples:
            return all_audio

        # Return the most recent samples
        return all_audio[-target_samples:]

    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self._total_samples / self.sample_rate

    def has_minimum_audio(self) -> bool:
        """Check if buffer has enough audio for enrollment."""
        return self._total_samples >= self.min_samples

    def clear(self) -> None:
        """Clear the buffer."""
        self._chunks.clear()
        self._total_samples = 0

    def __len__(self) -> int:
        """Return total number of samples in buffer."""
        return self._total_samples

    def __bool__(self) -> bool:
        """Return True if buffer has any audio."""
        return self._total_samples > 0
