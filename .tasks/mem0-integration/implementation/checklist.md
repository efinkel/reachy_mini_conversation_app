# Implementation Checklist

## Completed Steps

- [x] **Step 1**: Add mem0ai dependency to pyproject.toml (as optional `memory` extra)
- [x] **Step 2**: Add MEM0_API_KEY and REACHY_USER_ID to config
- [x] **Step 3**: Create memory module with Mem0 client wrapper
- [x] **Step 4**: Create recall_memory tool
- [x] **Step 5**: Extend ToolDependencies with mem0 fields
- [x] **Step 6**: Wire up mem0 client in entry points
- [x] **Step 7**: Add transcript collection during conversations
- [x] **Step 8**: Implement auto-save on session shutdown
- [x] **Step 9**: Add recall_memory to default profile tools.txt
- [x] **Step 10**: Add memory instructions to system prompt

## Files Created

- `src/reachy_mini_conversation_app/memory/__init__.py`
- `src/reachy_mini_conversation_app/tools/recall_memory.py`
- `src/reachy_mini_conversation_app/prompts/memory_instructions.txt`

## Files Modified

- `pyproject.toml` - added `memory` optional dependency
- `.env.example` - added MEM0_API_KEY and REACHY_USER_ID
- `src/reachy_mini_conversation_app/config.py` - load mem0 config
- `src/reachy_mini_conversation_app/tools/core_tools.py` - extended ToolDependencies
- `src/reachy_mini_conversation_app/main.py` - wire up user_id and transcript
- `src/reachy_mini_conversation_app/openai_realtime.py` - transcript collection + auto-save
- `src/reachy_mini_conversation_app/profiles/default/tools.txt` - add recall_memory
- `src/reachy_mini_conversation_app/prompts/default_prompt.txt` - include memory instructions

## Testing Required

- [ ] Verify app runs without MEM0_API_KEY (graceful fallback)
- [ ] Test recall_memory tool returns "memory not available" without config
- [ ] Test with MEM0_API_KEY: conversation saves at session end
- [ ] Test recall_memory retrieves saved information in new session
