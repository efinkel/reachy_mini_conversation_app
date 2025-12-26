# Speaker Identification - Brainstorming Summary

## Problem Statement
Currently, Reachy identifies users via a static config value (`REACHY_USER_ID=eric`). We want Reachy to automatically recognize who is speaking based on their voice, enabling seamless multi-user support and more natural interactions.

## Solution Overview
Use Picovoice Eagle SDK for real-time speaker identification. Accept that the first ~3 seconds of speech may not have identification yet, so the first response may be generic before personalization kicks in.

## Key Decisions

### Approach: "Accept First-Response Delay"
- First response might be generic ("Hey there!")
- By second exchange, Reachy knows who's speaking
- Graceful degradation: if speaker unknown, continue without personalization

### Technology: Picovoice Eagle
- On-device speaker recognition (no cloud latency)
- Works on macOS, Linux, Raspberry Pi
- Free tier available
- ~3 seconds of audio needed for identification

## User Flow

### First-Time User (Enrollment)
```
User: "Hey Reachy, I'm Sarah"
Reachy: "Nice to meet you Sarah! Let me remember your voice."
       [Enrolls voice in background]
Reachy: "Got it! I'll recognize you next time."
```

### Returning User
```
User: "Hey Reachy"
       [Eagle: processing...]
Reachy: "Hey!" (generic - no ID yet)

User: "What did we talk about last time?"
       [Eagle: "Speaker: Sarah" confidence 0.91]
Reachy: "Sarah! Last time you mentioned your new garden project."
       [Memories now filtered to Sarah]
```

### Speaker Change Mid-Conversation
```
[Eric talking]
Reachy: responds with Eric's context

[Sarah starts talking]
       [Eagle detects speaker change]
Reachy: "Oh hey Sarah! Didn't realize you were here too."
       [Switches context to Sarah]
```

## Decisions Made

| Question | Decision |
|----------|----------|
| **Enrollment trigger** | Voice command ("Reachy, remember my voice, I'm [name]") |
| **Unknown speaker handling** | Ask "I don't recognize your voice - who am I talking to?" |
| **Confidence threshold** | 0.75 (balanced) |
| **Voiceprint storage** | `~/.reachy_mini/voiceprints/` |

## Edge Cases

**Multiple speakers simultaneously:**
- Eagle handles primary speaker detection
- Defer to last confident identification

## Architecture Sketch

```
┌─────────────────────────────────────────────────────────────────┐
│                     Audio Pipeline                               │
│                                                                  │
│  Mic ──► Audio Buffer ──┬──► Picovoice Eagle ──► Speaker ID     │
│                         │                            │           │
│                         │                            ▼           │
│                         │                    ┌─────────────┐     │
│                         │                    │ Speaker     │     │
│                         │                    │ Manager     │     │
│                         │                    │             │     │
│                         │                    │ • current_  │     │
│                         │                    │   speaker   │     │
│                         │                    │ • confidence│     │
│                         │                    │ • enrolled  │     │
│                         │                    │   profiles  │     │
│                         │                    └──────┬──────┘     │
│                         │                           │            │
│                         └──► OpenAI Realtime ◄──────┘            │
│                                    │                             │
│                                    ▼                             │
│                              ┌───────────┐                       │
│                              │ Response  │                       │
│                              │ + Context │                       │
│                              └───────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components Needed

1. **SpeakerManager** - Core class managing Eagle and speaker state
2. **VoiceprintStore** - Persist/load enrolled voiceprints
3. **Enrollment Tool** - LLM tool for voice enrollment
4. **Speaker Context Injection** - Update system prompt with current speaker
5. **Dynamic User ID** - Update `current_user_id` in ToolDependencies

## Resources
- [Picovoice Eagle SDK](https://picovoice.ai/platform/eagle/)
- [Eagle Python Quick Start](https://picovoice.ai/docs/quick-start/eagle-python/)
- [Eagle GitHub](https://github.com/Picovoice/eagle)
