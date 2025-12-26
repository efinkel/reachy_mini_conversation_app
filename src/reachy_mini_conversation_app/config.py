import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    # Mem0 memory integration (optional)
    MEM0_API_KEY = os.getenv("MEM0_API_KEY")
    REACHY_USER_ID = os.getenv("REACHY_USER_ID", "default")
    logger.debug(f"Mem0 configured: {bool(MEM0_API_KEY)}, User ID: {REACHY_USER_ID}")

    # Speaker identification (optional)
    VOICEPRINT_DIR = os.getenv("VOICEPRINT_DIR", os.path.expanduser("~/.reachy_mini/voiceprints"))
    SPEAKER_CONFIDENCE_THRESHOLD = float(os.getenv("SPEAKER_CONFIDENCE_THRESHOLD", "0.50"))
    SPEAKER_DEBUG_PLOTS = os.getenv("SPEAKER_DEBUG_PLOTS", "").lower() in ("true", "1", "yes")
    SPEAKER_MAX_BUFFER_DURATION = float(os.getenv("SPEAKER_MAX_BUFFER_DURATION", "900"))  # 15 min
    SPEAKER_MIN_CONVERSATION_DURATION = float(os.getenv("SPEAKER_MIN_CONVERSATION_DURATION", "60"))  # 1 min

    # Embedding extraction
    SPEAKER_EMBEDDING_WINDOW_SECONDS = float(os.getenv("SPEAKER_EMBEDDING_WINDOW_SECONDS", "6.0"))

    # Background diarization
    SPEAKER_DIARIZATION_WINDOW_SECONDS = float(os.getenv("SPEAKER_DIARIZATION_WINDOW_SECONDS", "180.0"))  # 3 min
    SPEAKER_DIARIZATION_COOLDOWN_SECONDS = float(os.getenv("SPEAKER_DIARIZATION_COOLDOWN_SECONDS", "30.0"))

    # Online adaptation (with safeguards)
    SPEAKER_ADAPTATION_ALPHA = float(os.getenv("SPEAKER_ADAPTATION_ALPHA", "0.1"))
    SPEAKER_ADAPTATION_MIN_CONFIDENCE = float(os.getenv("SPEAKER_ADAPTATION_MIN_CONFIDENCE", "0.70"))
    SPEAKER_ADAPTATION_COOLDOWN_SECONDS = float(os.getenv("SPEAKER_ADAPTATION_COOLDOWN_SECONDS", "60.0"))
    SPEAKER_ADAPTATION_MAX_PER_SESSION = int(os.getenv("SPEAKER_ADAPTATION_MAX_PER_SESSION", "10"))

    # Session tracking
    SPEAKER_SESSION_SIMILARITY_THRESHOLD = float(os.getenv("SPEAKER_SESSION_SIMILARITY_THRESHOLD", "0.55"))
    SPEAKER_MIN_EMBEDDINGS_FOR_ENROLLMENT = int(os.getenv("SPEAKER_MIN_EMBEDDINGS_FOR_ENROLLMENT", "3"))

    logger.debug(f"Speaker identification: voiceprint_dir={VOICEPRINT_DIR}, threshold={SPEAKER_CONFIDENCE_THRESHOLD}, debug_plots={SPEAKER_DEBUG_PLOTS}")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
