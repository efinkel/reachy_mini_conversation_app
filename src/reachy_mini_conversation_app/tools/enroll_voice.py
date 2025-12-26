"""Tool to enroll or update a speaker's voice for identification."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class EnrollVoice(Tool):
    """Enroll or update a speaker's voice for identification."""

    name = "enroll_voice"
    description = (
        "Enroll or update a person's voice so you can recognize them. "
        "Use this when someone asks you to: remember their voice, update their voiceprint, "
        "or re-enroll. After calling this, the person should keep talking for about 5 seconds."
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
        """Start voice enrollment for a speaker."""
        name = (kwargs.get("name") or "").strip()
        if not name:
            return {"error": "name must be a non-empty string"}

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

        if speaker_manager.is_enrolling:
            return {
                "success": False,
                "message": "Already enrolling a speaker. Please wait.",
            }

        # Check if this is an update
        is_update = name.lower().strip() in [s.lower() for s in speaker_manager.list_enrolled_speakers()]

        # Start enrollment - audio collection happens in the audio pipeline
        started = speaker_manager.start_enrollment(name)

        if started:
            action = "Updating" if is_update else "Starting"
            return {
                "success": True,
                "message": f"{action} voice enrollment for {name}. Keep talking for about 5 seconds.",
                "enrolling": True,
                "is_update": is_update,
            }
        else:
            return {
                "success": False,
                "message": "Failed to start enrollment.",
            }
