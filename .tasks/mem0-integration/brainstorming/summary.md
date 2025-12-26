# Mem0 Integration - Brainstorming Summary

## Problem Statement
Reachy Mini currently has no persistent memory across conversation sessions. Each interaction starts fresh, making it impossible to build personal relationships or remember user preferences.

## Solution
Integrate Mem0 as a memory layer to give Reachy persistent, semantic memory that:
- Remembers facts about people (names, preferences, interests)
- Stores conversation history summaries
- Retrieves relevant context during conversations

## Key Decisions

### Memory Scope
- **Facts about people**: Names, preferences, interests, relationships
- **Conversation summaries**: Key topics discussed, important events mentioned

### User Identification (Phase 1)
- **Voice-based identification**: Reachy asks "Who am I talking to?" at session start
- Simple and requires no additional infrastructure
- Future: Face recognition for automatic identification

### Memory Access Pattern
- **Tool-based**: LLM calls a `recall_memory` tool to retrieve relevant memories
- Gives the model control over when to access memory
- More transparent than hidden system prompt injection

### Storage Backend
- **Mem0 Cloud API**: Managed service, no local infrastructure needed
- Requires MEM0_API_KEY in environment

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Conversation Flow                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Speech ──► Transcription ──► OpenAI Realtime      │
│                                          │               │
│                                          ▼               │
│                                    ┌─────────────┐       │
│                                    │ Tool Calls  │       │
│                                    └─────────────┘       │
│                                          │               │
│                    ┌─────────────────────┼───────────┐   │
│                    ▼                     ▼           ▼   │
│              ┌──────────┐         ┌──────────┐  [other]  │
│              │ save_    │         │ recall_  │           │
│              │ memory   │         │ memory   │           │
│              └────┬─────┘         └────┬─────┘           │
│                   │                    │                 │
│                   ▼                    ▼                 │
│              ┌─────────────────────────────┐             │
│              │      Mem0 Cloud API         │             │
│              │  (Vector DB + LLM Extract)  │             │
│              └─────────────────────────────┘             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Open Questions
1. How should user_id be formatted? (simple name string?)
2. Should we add automatic memory saving after each exchange, or only when tool is called?
3. What prompting helps Reachy naturally use the memory tools?

## Resources
- [Mem0 Docs](https://docs.mem0.ai/)
- [Mem0 Python Quickstart](https://docs.mem0.ai/open-source/python-quickstart)
- [Mem0 Cloud API](https://mem0.ai/)
- [Conversation App Repo](https://github.com/pollen-robotics/reachy_mini_conversation_app)
