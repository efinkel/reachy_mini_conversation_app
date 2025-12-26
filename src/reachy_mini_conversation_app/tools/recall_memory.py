"""Tool to recall memories about the current user."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.memory import search_memories


logger = logging.getLogger(__name__)


class RecallMemory(Tool):
    """Recall memories about the person you're talking to."""

    name = "recall_memory"
    description = (
        "Search your memory for information about the person you're talking to. "
        "Use this to remember their name, preferences, past conversations, or any "
        "personal details they've shared before."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to search for in memory. Examples: "
                    "'their name', 'food preferences', 'what we talked about last time', "
                    "'their hobbies', 'their job'"
                ),
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Search memories for relevant information."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}

        # Get user_id from deps if available, otherwise use default
        user_id = getattr(deps, "current_user_id", None)
        logger.info(f"Tool call: recall_memory query='{query}', user_id='{user_id}'")

        # Debug: Check how many memories exist for this user
        try:
            from reachy_mini_conversation_app.memory import get_all_memories
            all_memories = get_all_memories(user_id=user_id)
            logger.info(f"Total memories for user '{user_id}': {len(all_memories)}")
            if all_memories:
                logger.debug(f"Sample memory: {all_memories[0].get('memory', 'N/A')[:100]}")
        except Exception as e:
            logger.debug(f"Debug get_all failed: {e}")

        results = search_memories(query, user_id=user_id)

        # Handle both list and dict response formats
        if isinstance(results, dict):
            results = results.get("results", [])

        if not results:
            return {
                "memories": [],
                "message": "No memories found for this query.",
            }

        # Format memories for the LLM
        formatted_memories = []
        for result in results[:5]:  # Limit to top 5 results
            memory_text = result.get("memory", "")
            if memory_text:
                formatted_memories.append(memory_text)

        return {
            "memories": formatted_memories,
            "message": f"Found {len(formatted_memories)} relevant memories.",
        }
