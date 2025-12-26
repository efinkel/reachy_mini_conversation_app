# Wake Word Detection

**Status**: Planning
**Goal**: Add "Hey Reachy" wake word activation to reduce always-on listening and API costs
**Created**: 2024-12-26

---

## Overview

Currently the conversation app is always listening and streaming to OpenAI Realtime API. This:
- Uses API credits continuously
- May feel intrusive (always listening)
- Doesn't have a natural "conversation start" signal

Adding wake word detection would allow idle/active states with clear activation.

---

## Design Options

### Option A: Always Listening, Wake Word Triggers Response

```
┌─────────────────────────────────────────────────┐
│ Idle (listening for wake word only)             │
│   - Wake word model running locally             │
│   - Speaker ID can run in background            │
│   - No OpenAI connection                        │
│         ↓ "Hey Reachy"                          │
│ Active (full conversation mode)                 │
│   - OpenAI Realtime connected                   │
│   - Full audio streaming                        │
│   - Visual/audio acknowledgment                 │
│         ↓ Timeout or "goodbye"                  │
│ Back to Idle                                    │
└─────────────────────────────────────────────────┘
```

### Option B: Push-to-Talk Alternative

- Physical button or keyboard shortcut
- Simpler, no always-on processing
- May be preferred for some use cases

### Option C: Hybrid

- Wake word OR button activation
- Best flexibility

---

## Wake Word Technology Options

### 1. Porcupine (Picovoice)
- **Pros**: High accuracy, custom wake words, well-documented
- **Cons**: Commercial (free tier has limits), requires API key
- **Custom wake word**: Yes, via console training
- **Platforms**: Cross-platform, runs on-device

### 2. OpenWakeWord
- **Pros**: Open source, runs locally, no API key
- **Cons**: Less mature, may need training for custom words
- **Custom wake word**: Yes, with training
- **GitHub**: https://github.com/dscripka/openWakeWord

### 3. Snowboy (Legacy)
- **Pros**: Was popular, runs locally
- **Cons**: No longer maintained (Kitt.ai shut down)
- **Status**: Not recommended for new projects

### 4. Whisper-based Keyword Spotting
- **Pros**: Already have Whisper, high accuracy
- **Cons**: Heavier compute, may have latency
- **Approach**: Run small Whisper model, look for specific phrases

### 5. Vosk with Keyword Spotting
- **Pros**: Open source, lightweight
- **Cons**: Less accurate for wake words specifically
- **Approach**: Small vocabulary ASR tuned for wake phrase

---

## TODO List

### Research Phase

- [ ] Test Porcupine with default wake words
- [ ] Test OpenWakeWord with default models
- [ ] Compare latency and accuracy
- [ ] Evaluate custom wake word training difficulty

### Design Phase

- [ ] Choose activation flow (A, B, or C)
- [ ] Design state machine for idle/active transitions
- [ ] Define timeout behavior and "goodbye" phrases
- [ ] Plan visual/audio feedback for state changes

### Implementation Phase

- [ ] Integrate chosen wake word library
- [ ] Add idle/active state management to console.py
- [ ] Implement OpenAI connection lifecycle (connect on wake, disconnect on idle)
- [ ] Add acknowledgment sound/animation on wake
- [ ] Add timeout logic (return to idle after N seconds of silence)
- [ ] Handle "goodbye" or "sleep" commands

### Polish

- [ ] Custom wake word training (if using trainable solution)
- [ ] User configuration for wake phrase
- [ ] Visual indicator of current state (LED, UI, robot expression)
- [ ] Graceful handling of missed wake words

---

## Key Integration Points

| File | Changes Needed |
|------|----------------|
| `console.py` | State machine, wake word callback, connection lifecycle |
| `openai_realtime.py` | Connect/disconnect methods for idle/active |
| `config.py` | Wake word settings, timeout values |
| New: `wake_word/` | Wake word detection module |

---

## Notes

- Wake word should run continuously with minimal CPU
- Speaker ID could potentially keep running in idle (to pre-identify before response)
- Consider: should identified speaker affect response? "Hey Reachy" → "Hi Eric, what's up?"
- Battery/power considerations for always-on detection
