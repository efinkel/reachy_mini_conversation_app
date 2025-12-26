# Mem0 Integration - Technical Plan

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Memory types | User Memory only | Persistent per-person facts, skip session/org |
| Save trigger | Auto-save at session end | Summarize conversation, no explicit tool needed |
| Retrieval | `recall_memory` tool | LLM decides when to query for context |
| User ID | Voice-based (future), config for now | Start simple with configurable default |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Conversation Session                        │
│                                                              │
│   User speaks ──► OpenAI Realtime ──► Response              │
│                         │                                    │
│                         ▼                                    │
│                  ┌─────────────┐                             │
│                  │ Tool Call?  │                             │
│                  └──────┬──────┘                             │
│                         │                                    │
│            ┌────────────┼────────────┐                       │
│            ▼            ▼            ▼                       │
│      [dance]    [recall_memory]  [camera] ...               │
│                        │                                     │
│                        ▼                                     │
│                 ┌─────────────┐                              │
│                 │ Mem0 Cloud  │◄─── search(query, user_id)  │
│                 └─────────────┘                              │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                    Session End                               │
│                                                              │
│   Transcript ──► Summarize ──► mem0.add(summary, user_id)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Add mem0ai dependency
- Add `mem0ai` to `pyproject.toml` dependencies
- Run `uv sync` to install

### Step 2: Add configuration
- Add `MEM0_API_KEY` to `config.py`
- Add `REACHY_USER_ID` for default user identification
- Update `.env.example` with placeholders

### Step 3: Create memory module
- Create `src/reachy_mini_conversation_app/memory/__init__.py`
- Lazy-init MemoryClient when API key present
- Expose `search_memories(query, user_id)` and `save_memories(messages, user_id)`

### Step 4: Create recall_memory tool
- Create `src/reachy_mini_conversation_app/tools/recall_memory.py`
- Parameters: `query` (string) - what to remember about
- Returns: list of relevant memories as text
- Uses `current_user_id` from ToolDependencies

### Step 5: Update ToolDependencies
- Add `mem0_client` (optional, None if no API key)
- Add `current_user_id: str` (from config or session)
- Add `conversation_transcript: list` (for end-of-session save)

### Step 6: Wire up in entry points
- Initialize mem0 client in `headless_personality.py`
- Initialize in `gradio_personality.py`
- Pass to ToolDependencies

### Step 7: Add transcript collection
- Capture user/assistant messages during session
- Store in ToolDependencies or handler state

### Step 8: Implement session-end auto-save
- Hook into `shutdown()` in `OpenaiRealtimeHandler`
- Summarize transcript and save to Mem0
- Handle graceful degradation if API unavailable

### Step 9: Add memory tools to profiles
- Add `recall_memory` to `profiles/default/tools.txt`
- (Other profiles can opt-in as needed)

### Step 10: Update prompts
- Create `prompts/memory_instructions.txt` snippet
- Explain to LLM: "You can recall memories about the person you're talking to"

## File Changes

| File | Change |
|------|--------|
| `pyproject.toml` | Add `mem0ai` dependency |
| `.env.example` | Add `MEM0_API_KEY`, `REACHY_USER_ID` |
| `config.py` | Load new env vars |
| `memory/__init__.py` | New - client wrapper |
| `tools/recall_memory.py` | New - recall tool |
| `tools/core_tools.py` | Extend ToolDependencies |
| `openai_realtime.py` | Collect transcript, auto-save on shutdown |
| `headless_personality.py` | Wire up mem0 client |
| `gradio_personality.py` | Wire up mem0 client |
| `profiles/default/tools.txt` | Add `recall_memory` |
| `prompts/memory_instructions.txt` | New - prompt snippet |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| No API key configured | Graceful fallback, tool returns "memory not available" |
| API latency on shutdown | Async fire-and-forget with timeout |
| Empty transcripts | Skip save if nothing meaningful said |
| Sensitive info in memory | Mem0 handles storage; document privacy implications |

## Test Plan

1. **No API key**: Verify app runs, tool returns graceful error
2. **recall_memory**: Mock API, verify tool returns formatted memories
3. **Auto-save**: Verify shutdown triggers save with transcript summary
4. **End-to-end**: Manual test - have conversation, restart, recall works
