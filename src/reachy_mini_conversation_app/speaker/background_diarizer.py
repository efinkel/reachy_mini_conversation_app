"""Background diarization worker for non-blocking speaker analysis."""

import logging
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


def _diarization_worker(
    audio: np.ndarray,
    sample_rate: int,
    result_queue: mp.Queue,
) -> None:
    """Worker function that runs diarization in a separate process.

    Args:
        audio: Audio samples to process
        sample_rate: Sample rate in Hz
        result_queue: Queue to put results into
    """
    try:
        from reachy_mini_conversation_app.speaker.diarization import run_diarization

        segments = run_diarization(audio, sample_rate)
        result_queue.put({
            "success": True,
            "segments": segments,
            "audio_duration": len(audio) / sample_rate,
        })
    except Exception as e:
        logger.error(f"Diarization worker error: {e}")
        result_queue.put({
            "success": False,
            "error": str(e),
        })


class BackgroundDiarizer:
    """Manages background diarization with worker pool.

    Runs speaker diarization in a separate process to avoid blocking
    the main conversation. Uses a worker pool with max 1 active process
    and 1 pending request to prevent accumulation.
    """

    def __init__(
        self,
        window_seconds: float = 180.0,  # 3 minutes
        cooldown_seconds: float = 30.0,
        on_results: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize the background diarizer.

        Args:
            window_seconds: How much audio to process when triggered (seconds)
            cooldown_seconds: Minimum time between triggers
            on_results: Optional callback when results are ready
        """
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.on_results = on_results

        self._process: Optional[mp.Process] = None
        self._result_queue: mp.Queue = mp.Queue()
        self._pending_audio: Optional[np.ndarray] = None
        self._last_trigger_time: float = 0
        self._active_audio_duration: float = 0

    def trigger(self, audio_buffer: Any, force: bool = False) -> bool:
        """Trigger background diarization if conditions are met.

        Args:
            audio_buffer: AudioBuffer instance to get audio from
            force: If True, skip cooldown check

        Returns:
            True if diarization was triggered, False otherwise
        """
        now = time.time()

        # Cooldown check (unless forced)
        if not force and (now - self._last_trigger_time) < self.cooldown_seconds:
            logger.debug(f"Diarization cooldown active ({now - self._last_trigger_time:.1f}s < {self.cooldown_seconds}s)")
            return False

        # Get recent audio window
        audio = audio_buffer.get_recent(self.window_seconds)
        if audio is None:
            logger.debug("No audio available for diarization")
            return False

        min_samples = int(self.window_seconds * 0.3 * 16000)  # At least 30% of window
        if len(audio) < min_samples:
            logger.debug(f"Not enough audio for diarization ({len(audio)} < {min_samples})")
            return False

        # If worker busy, queue this request (replacing any pending)
        if self._process is not None and self._process.is_alive():
            logger.debug("Diarization worker busy, queuing request")
            self._pending_audio = audio
            return False

        # Start worker
        self._start_worker(audio)
        self._last_trigger_time = now
        return True

    def check_results(self) -> Optional[Dict[str, Any]]:
        """Non-blocking check for completed results.

        Also handles starting pending requests when worker finishes.

        Returns:
            Result dict if available, None otherwise
        """
        # Check if worker finished
        if self._process is not None and not self._process.is_alive():
            self._process.join(timeout=0.1)
            self._process = None
            logger.debug("Background diarization worker finished")

            # Process any pending request
            if self._pending_audio is not None:
                logger.info("Starting queued diarization request")
                self._start_worker(self._pending_audio)
                self._pending_audio = None

        # Check result queue (non-blocking)
        try:
            result = self._result_queue.get_nowait()
            if result.get("success"):
                logger.info(
                    f"Background diarization complete: "
                    f"{len(result.get('segments', []))} segments in {result.get('audio_duration', 0):.1f}s audio"
                )
            else:
                logger.warning(f"Background diarization failed: {result.get('error')}")

            if self.on_results is not None:
                self.on_results(result)
            return result
        except Exception:
            return None

    def _start_worker(self, audio: np.ndarray) -> None:
        """Start the background diarization process.

        Args:
            audio: Audio samples to process
        """
        self._active_audio_duration = len(audio) / 16000
        logger.info(f"Starting background diarization on {self._active_audio_duration:.1f}s of audio")

        self._process = mp.Process(
            target=_diarization_worker,
            args=(audio, 16000, self._result_queue),
            daemon=True,
        )
        self._process.start()

    def is_busy(self) -> bool:
        """Check if diarization is currently running."""
        return self._process is not None and self._process.is_alive()

    def has_pending(self) -> bool:
        """Check if there's a pending request queued."""
        return self._pending_audio is not None

    def stop(self) -> None:
        """Stop any running diarization and clean up."""
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None
        self._pending_audio = None

        # Drain result queue
        while True:
            try:
                self._result_queue.get_nowait()
            except Exception:
                break
