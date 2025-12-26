"""Memory module for Mem0 integration."""

import logging
from typing import Any

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

# Lazy-loaded client
_client: Any = None


def get_mem0_client() -> Any:
    """Get or create the Mem0 client.

    Returns None if MEM0_API_KEY is not configured.
    """
    global _client

    if _client is not None:
        return _client

    if not config.MEM0_API_KEY:
        logger.debug("MEM0_API_KEY not configured, memory features disabled")
        return None

    try:
        from mem0 import MemoryClient

        _client = MemoryClient(api_key=config.MEM0_API_KEY)
        logger.info("Mem0 client initialized successfully")
        return _client
    except ImportError:
        logger.warning("mem0ai package not installed, memory features disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Mem0 client: {e}")
        return None


def search_memories(query: str, user_id: str | None = None) -> list[dict[str, Any]]:
    """Search for relevant memories.

    Args:
        query: The search query
        user_id: Optional user ID filter (defaults to config.REACHY_USER_ID)

    Returns:
        List of memory results with 'memory' and 'score' keys
    """
    client = get_mem0_client()
    if client is None:
        return []

    user_id = user_id or config.REACHY_USER_ID

    try:
        # Mem0 API requires filters in AND/OR format
        filters = {"AND": [{"user_id": user_id}]}
        results = client.search(query, filters=filters)
        logger.debug(f"Memory search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        return []


def save_memories(messages: list[dict[str, str]], user_id: str | None = None) -> bool:
    """Save conversation messages to memory.

    Uses Contextual Add v2 which automatically retrieves context from
    previous conversations for better memory deduplication and merging.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        user_id: Optional user ID (defaults to config.REACHY_USER_ID)

    Returns:
        True if save succeeded, False otherwise
    """
    client = get_mem0_client()
    if client is None:
        return False

    if not messages:
        logger.debug("No messages to save to memory")
        return False

    user_id = user_id or config.REACHY_USER_ID

    try:
        # Use v2 for contextual add - better deduplication and memory merging
        client.add(messages, user_id=user_id, version="v2")
        logger.info(f"Saved {len(messages)} messages to memory for user '{user_id}' (v2)")
        return True
    except Exception as e:
        logger.error(f"Failed to save memories: {e}")
        return False
