"""Post-conversation speaker diarization and enrollment."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.speaker.voiceprint_store import VoiceprintStore


logger = logging.getLogger(__name__)

# Lazy-loaded pipeline
_diarization_pipeline = None


def get_diarization_pipeline():
    """Get or create the diarization pipeline."""
    global _diarization_pipeline

    if _diarization_pipeline is not None:
        return _diarization_pipeline

    if not config.HF_TOKEN:
        logger.warning("HF_TOKEN not set, diarization disabled")
        return None

    try:
        from pyannote.audio import Pipeline

        logger.info("Loading diarization pipeline...")
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=config.HF_TOKEN
        )
        logger.info("Diarization pipeline loaded")
        return _diarization_pipeline
    except ImportError:
        logger.warning("pyannote.audio not installed, diarization disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}")
        return None


def run_diarization(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> Optional[List[Tuple[float, float, str]]]:
    """Run speaker diarization on audio.

    Args:
        audio: Audio samples (mono, float32)
        sample_rate: Sample rate in Hz

    Returns:
        List of (start_time, end_time, speaker_label) tuples, or None on failure
    """
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return None

    try:
        import torch

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

        logger.info("Running diarization...")
        result = pipeline(audio_dict)
        diarization = result.speaker_diarization

        segments = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            segments.append((segment.start, segment.end, label))

        logger.info(f"Diarization found {len(set(s[2] for s in segments))} speakers, {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None


def extract_speaker_segments(
    audio: np.ndarray,
    segments: List[Tuple[float, float, str]],
    speaker_label: str,
    sample_rate: int = 16000,
    min_segment_duration: float = 1.0,
    max_segments: int = 10,
) -> List[np.ndarray]:
    """Extract audio segments for a specific speaker.

    Args:
        audio: Full audio array
        segments: Diarization segments (start, end, label)
        speaker_label: Which speaker to extract
        sample_rate: Sample rate in Hz
        min_segment_duration: Minimum segment duration to include
        max_segments: Maximum number of segments to return

    Returns:
        List of audio segments for the speaker
    """
    speaker_segments = []

    for start, end, label in segments:
        if label != speaker_label:
            continue
        if end - start < min_segment_duration:
            continue

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_audio = audio[start_sample:end_sample]
        speaker_segments.append((end - start, segment_audio))

    # Sort by duration (longest first) and take top N
    speaker_segments.sort(key=lambda x: x[0], reverse=True)
    return [seg[1] for seg in speaker_segments[:max_segments]]


def match_speakers_to_names(
    segments: List[Tuple[float, float, str]],
    mentioned_names: Dict[str, float],  # name -> timestamp when mentioned
    audio: np.ndarray,
    sample_rate: int = 16000,
    store: Optional[VoiceprintStore] = None,
) -> Dict[str, str]:
    """Match diarized speaker labels to names.

    Strategy (in order):
    1. Compare speaker embeddings to existing voiceprints
    2. If a name is mentioned, match speaker talking at that time
    3. Unmatched speakers remain unidentified

    Args:
        segments: Diarization segments
        mentioned_names: Dict of names to timestamps when they were mentioned
        audio: Full audio array
        sample_rate: Sample rate
        store: Voiceprint store for existing comparisons

    Returns:
        Dict mapping speaker_label -> name
    """
    speaker_to_name: Dict[str, str] = {}

    # Get unique speaker labels
    unique_speakers = sorted(set(s[2] for s in segments))

    # Strategy 1: Match against existing voiceprints
    if store is not None:
        enrolled = store.get_all()
        if enrolled:
            try:
                import torch
                from pyannote.audio import Model, Inference
                from scipy.spatial.distance import cosine

                model = Model.from_pretrained("pyannote/embedding", token=config.HF_TOKEN)
                inference = Inference(model, window="whole")

                for speaker_label in unique_speakers:
                    # Get a representative segment for this speaker
                    speaker_segments = extract_speaker_segments(
                        audio, segments, speaker_label, sample_rate,
                        min_segment_duration=2.0, max_segments=3
                    )
                    if not speaker_segments:
                        continue

                    # Get embedding from longest segment
                    seg_audio = speaker_segments[0]
                    waveform = torch.from_numpy(seg_audio.astype(np.float32)).unsqueeze(0)
                    embedding = np.array(inference({"waveform": waveform, "sample_rate": sample_rate}))

                    # Compare to enrolled speakers
                    best_match = None
                    best_similarity = 0.0
                    for name, stored_emb in enrolled.items():
                        similarity = 1 - cosine(embedding, stored_emb)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name

                    if best_match and best_similarity >= config.SPEAKER_CONFIDENCE_THRESHOLD:
                        speaker_to_name[speaker_label] = best_match
                        logger.info(f"Matched {speaker_label} to '{best_match}' via voiceprint ({best_similarity:.0%})")

            except Exception as e:
                logger.warning(f"Voiceprint matching failed: {e}")

    # Strategy 2: Match by mentioned names (for speakers not yet matched)
    if mentioned_names:
        for name, timestamp in mentioned_names.items():
            name_lower = name.lower()
            # Skip if this name is already assigned
            if name_lower in speaker_to_name.values():
                continue

            # Find the speaker active around this timestamp
            for start, end, label in segments:
                if start <= timestamp <= end + 5.0:
                    if label not in speaker_to_name:
                        speaker_to_name[label] = name_lower
                        logger.info(f"Matched {label} to '{name}' based on timing")
                    break

    # Log unmatched speakers
    unmatched = [s for s in unique_speakers if s not in speaker_to_name]
    if unmatched:
        logger.info(f"Unmatched speakers (will not be enrolled): {unmatched}")

    return speaker_to_name


def process_conversation_end(
    audio: np.ndarray,
    mentioned_names: Dict[str, float],
    sample_rate: int = 16000,
    store: Optional[VoiceprintStore] = None,
) -> Dict[str, bool]:
    """Process audio at conversation end for speaker enrollment.

    Args:
        audio: Full conversation audio
        mentioned_names: Names mentioned and when (name -> timestamp)
        sample_rate: Audio sample rate
        store: Voiceprint store (created if None)

    Returns:
        Dict of name -> success for each enrollment attempt
    """
    if store is None:
        store = VoiceprintStore()

    results: Dict[str, bool] = {}

    # Check minimum duration
    duration = len(audio) / sample_rate
    if duration < config.SPEAKER_MIN_CONVERSATION_DURATION:
        logger.info(f"Conversation too short ({duration:.1f}s < {config.SPEAKER_MIN_CONVERSATION_DURATION}s), skipping enrollment")
        return results

    # Run diarization
    segments = run_diarization(audio, sample_rate)
    if segments is None:
        logger.warning("Diarization failed, skipping enrollment")
        return results

    # Match speakers to names
    speaker_to_name = match_speakers_to_names(
        segments, mentioned_names, audio, sample_rate, store
    )

    if not speaker_to_name:
        logger.info("No speakers matched to names, skipping enrollment")
        # Generate debug plot anyway if enabled
        if config.SPEAKER_DEBUG_PLOTS:
            _generate_debug_plot(audio, segments, {}, sample_rate)
        return results

    # Extract embeddings and enroll
    try:
        from pyannote.audio import Model, Inference

        model = Model.from_pretrained("pyannote/embedding", token=config.HF_TOKEN)
        inference = Inference(model, window="whole")

        for speaker_label, name in speaker_to_name.items():
            try:
                # Get segments for this speaker
                speaker_audio_segments = extract_speaker_segments(
                    audio, segments, speaker_label, sample_rate
                )

                if len(speaker_audio_segments) < 2:
                    logger.warning(f"Not enough segments for {name}, skipping")
                    results[name] = False
                    continue

                # Extract embeddings from each segment
                import torch
                embeddings = []
                for seg_audio in speaker_audio_segments:
                    waveform = torch.from_numpy(seg_audio.astype(np.float32)).unsqueeze(0)
                    emb = inference({"waveform": waveform, "sample_rate": sample_rate})
                    embeddings.append(np.array(emb))

                # Average embeddings
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                # Check if we should update existing or create new
                existing = store.get(name)
                if existing is not None:
                    # Blend with existing (weighted average favoring new)
                    blended = 0.3 * existing + 0.7 * avg_embedding
                    blended = blended / np.linalg.norm(blended)
                    store.save(name, blended)
                    logger.info(f"Updated voiceprint for '{name}' with {len(embeddings)} segments")
                else:
                    store.save(name, avg_embedding)
                    logger.info(f"Created voiceprint for '{name}' with {len(embeddings)} segments")

                results[name] = True

            except Exception as e:
                logger.error(f"Failed to enroll {name}: {e}")
                results[name] = False

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")

    # Generate debug plot if enabled
    if config.SPEAKER_DEBUG_PLOTS:
        _generate_debug_plot(audio, segments, speaker_to_name, sample_rate)

    return results


def _generate_debug_plot(
    audio: np.ndarray,
    segments: List[Tuple[float, float, str]],
    speaker_to_name: Dict[str, str],
    sample_rate: int = 16000,
) -> Optional[Path]:
    """Generate debug visualization of diarization."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        duration = len(audio) / sample_rate

        # Get unique speakers
        speakers = sorted(set(s[2] for s in segments))
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(speakers), 2)))
        speaker_colors = {spk: colors[i] for i, spk in enumerate(speakers)}

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), height_ratios=[1, 2])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f'Speaker Diarization - {timestamp}', fontsize=14)

        # Waveform
        ax1 = axes[0]
        time_axis = np.linspace(0, duration, len(audio))
        ax1.plot(time_axis, audio, linewidth=0.5, color='steelblue', alpha=0.7)
        ax1.set_xlim(0, duration)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Audio Waveform')
        ax1.grid(True, alpha=0.3)

        # Speaker timeline
        ax2 = axes[1]
        ax2.set_xlim(0, duration)
        ax2.set_ylim(-0.5, len(speakers) - 0.5)

        # Labels with names if matched
        labels = []
        for spk in speakers:
            if spk in speaker_to_name:
                labels.append(f"{spk} ({speaker_to_name[spk]})")
            else:
                labels.append(spk)

        ax2.set_yticks(range(len(speakers)))
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Speaker')
        ax2.set_title('Speaker Timeline')
        ax2.grid(True, alpha=0.3, axis='x')

        # Draw segments
        for start, end, label in segments:
            speaker_idx = speakers.index(label)
            rect = Rectangle(
                (start, speaker_idx - 0.4),
                end - start,
                0.8,
                facecolor=speaker_colors[label],
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
            ax2.add_patch(rect)

        plt.tight_layout()

        # Save
        output_dir = Path(config.VOICEPRINT_DIR) / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"diarization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Debug plot saved to {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to generate debug plot: {e}")
        return None
