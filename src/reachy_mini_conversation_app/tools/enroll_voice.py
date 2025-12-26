"""Tool to enroll a speaker's voice for identification."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class EnrollVoice(Tool):
    """Enroll a speaker's voice for future identification."""

    name = "enroll_voice"
    description = (
        "Enroll a person's voice so you can recognize them in the future. "
        "Use this when someone asks you to remember their voice. "
        "After calling this, the person should keep talking for a few seconds "
        "to capture enough audio for their voiceprint."
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

        # Start enrollment - audio collection happens in the audio pipeline
        started = speaker_manager.start_enrollment(name)

        if started:
            return {
                "success": True,
                "message": f"Starting voice enrollment for {name}. Keep talking for about 5 seconds.",
                "enrolling": True,
            }
        else:
            return {
                "success": False,
                "message": "Failed to start enrollment.",
            }
