"""Tool to enroll or update a speaker's voice for identification."""

import logging
import time
from typing import Any, Dict

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class EnrollVoice(Tool):
    """Enroll or update a speaker's voice for identification via diarization."""

    name = "enroll_voice"
    description = (
        "Enroll or update a person's voice so you can recognize them. "
        "Use this when someone asks you to: remember their voice, update their voiceprint, "
        "or re-enroll. After calling this, ask them to keep talking naturally for about 20-30 seconds."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the person to enroll (e.g., 'Sarah', 'Eric')",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Queue voice enrollment for a speaker via diarization.

        Enrollment happens through background diarization which provides
        cleaner audio segments for better voiceprint quality.
        """
        name = (kwargs.get("name") or "").strip()
        if not name:
            return {"error": "name must be a non-empty string"}

        name_lower = name.lower().strip()
        logger.info(f"Tool call: enroll_voice name='{name}'")

        speaker_manager = getattr(deps, "speaker_manager", None)
        if speaker_manager is None:
            return {
                "success": False,
                "message": "Speaker identification is not available.",
            }

        if not speaker_manager._initialized:
            return {
                "success": False,
                "message": "Speaker identification is not initialized. Please check HF_TOKEN.",
            }

        # Check if this is an update to existing voiceprint
        enrolled_speakers = [s.lower() for s in speaker_manager.list_enrolled_speakers()]
        is_update = name_lower in enrolled_speakers

        # For updates, remove old voiceprint so diarization creates a fresh one
        if is_update:
            speaker_manager.remove_speaker(name_lower)
            logger.info(f"Removed old voiceprint for '{name}' for re-enrollment")

        # Store in mentioned_names so diarization knows to enroll this speaker
        mentioned_names = getattr(deps, "mentioned_names", None)
        if mentioned_names is not None:
            mentioned_names[name_lower] = time.time()
            logger.info(f"Added '{name_lower}' to mentioned_names for diarization enrollment")

        # Trigger diarization if available
        audio_buffer = getattr(deps, "audio_buffer", None)
        diarization_triggered = False
        if speaker_manager.background_diarizer is not None and audio_buffer is not None:
            diarization_triggered = speaker_manager.background_diarizer.trigger(audio_buffer)
            if diarization_triggered:
                logger.info("Triggered background diarization for enrollment")

        action = "Re-enrolling" if is_update else "Enrolling"
        min_seconds = int(config.SPEAKER_MIN_ENROLLMENT_SECONDS)

        return {
            "success": True,
            "message": (
                f"{action} {name}. Keep talking naturally for about {min_seconds} seconds "
                f"and I'll learn your voice."
            ),
            "is_update": is_update,
            "diarization_triggered": diarization_triggered,
        }
