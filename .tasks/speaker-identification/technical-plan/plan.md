# Speaker Identification - Technical Plan (pyannote)

## Decisions Summary

| Decision | Choice |
|----------|--------|
| Library | pyannote.audio (speaker embedding + diarization) |
| Enrollment | **Post-conversation diarization** (automatic, no interruption) |
| Unknown handling | Ask who they are, label for later enrollment |
| Confidence threshold | 0.50 (cosine similarity) |
| Storage | `~/.reachy_mini/voiceprints/` |
| Min conversation | 60 seconds before enrolling |
| Max audio buffer | 15 minutes per session |
| Debugging | Generate diarization timeline plots |

## V2: Post-Conversation Enrollment (Diarization)

### Flow
```
1. Conversation starts
2. Audio buffered continuously (max 15 min rolling)
3. User mentions name → stored but no enrollment yet
4. Conversation continues naturally
5. Session ends (or 15 min mark):
   → Run speaker diarization
   → Match speakers to names mentioned
   → Extract embeddings from multiple segments
   → Average and save voiceprint
   → Generate debug plot
```

### Constraints
- Minimum 60s conversation before attempting enrollment
- Maximum 15 min audio buffer (rolling window)
- Debug plots saved to `~/.reachy_mini/voiceprints/debug/`

## pyannote API Overview

### Speaker Embedding
```python
from pyannote.audio import Model, Inference

# Load embedding model (requires HuggingFace token first time)
model = Model.from_pretrained("pyannote/embedding", use_auth_token="HF_TOKEN")
inference = Inference(model, window="whole")

# Get embedding from audio (in-memory)
embedding = inference({"waveform": waveform, "sample_rate": 16000})
# Returns: numpy array of shape (1, 512) or similar
```

### Comparing Speakers
```python
from scipy.spatial.distance import cosine

similarity = 1 - cosine(embedding1, embedding2)
# similarity > 0.75 = same speaker
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Audio Pipeline                                │
│                                                                      │
│  Microphone ──► Audio Buffer ──┬──► OpenAI Realtime (conversation)  │
│                                │                                     │
│                                └──► SpeakerManager                   │
│                                          │                           │
│                           ┌──────────────┴──────────────┐           │
│                           ▼                              ▼           │
│                    pyannote Inference           Enrollment Mode      │
│                    (get embedding)              (collect audio)      │
│                           │                              │           │
│                           ▼                              ▼           │
│                    Compare to enrolled         Save new embedding    │
│                    voiceprints                 to disk               │
│                           │                                          │
│                           ▼                                          │
│                    ┌─────────────┐                                   │
│                    │ Speaker ID  │                                   │
│                    │ + similarity│                                   │
│                    └──────┬──────┘                                   │
│                           │                                          │
│                           ▼                                          │
│                    Update current_user_id                            │
│                    + inject context to prompt                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Add pyannote dependency
- Add `pyannote.audio` to `pyproject.toml` (optional extra: `speaker`)
- Already tested - works on macOS with MPS acceleration

### Step 2: Add configuration
- `HF_TOKEN` in `.env` (HuggingFace token)
- `VOICEPRINT_DIR` defaulting to `~/.reachy_mini/voiceprints/`
- `SPEAKER_CONFIDENCE_THRESHOLD` defaulting to `0.75`

### Step 3: Create VoiceprintStore
- `src/reachy_mini_conversation_app/speaker/__init__.py`
- `src/reachy_mini_conversation_app/speaker/voiceprint_store.py`
- Load/save speaker embeddings as numpy arrays
- Map speaker names to embedding vectors

### Step 4: Create SpeakerManager
- `src/reachy_mini_conversation_app/speaker/speaker_manager.py`
- Initialize pyannote inference on startup
- Process audio chunks, accumulate ~3 seconds
- Compare embedding to enrolled speakers
- Track current speaker + confidence

### Step 5: Create enroll_voice tool
- `src/reachy_mini_conversation_app/tools/enroll_voice.py`
- LLM calls this when user says "remember my voice, I'm [name]"
- Parameters: `name` (string)
- Collects audio, extracts embedding, saves to disk

### Step 6: Integrate with audio pipeline
- Modify `console.py`
- Fork audio frames to SpeakerManager
- Update `deps.current_user_id` when speaker identified

### Step 7: Add speaker context injection
- When speaker changes, update system prompt context
- Auto-recall relevant memories for that user

### Step 8: Add tools to profile
- Add `enroll_voice` to `profiles/main/tools.txt`
- Update prompt with enrollment instructions

### Step 9: Handle unknown speakers
- When similarity < threshold for all enrolled speakers
- Inject prompt: "Ask who you're talking to"

## File Changes

| File | Change |
|------|--------|
| `pyproject.toml` | Add `pyannote.audio` optional dependency |
| `.env.example` | Add `HF_TOKEN`, `VOICEPRINT_DIR` |
| `config.py` | Load new config vars |
| `speaker/__init__.py` | New - package init |
| `speaker/voiceprint_store.py` | New - embedding persistence |
| `speaker/speaker_manager.py` | New - pyannote integration |
| `tools/enroll_voice.py` | New - enrollment tool |
| `tools/core_tools.py` | Add speaker fields to ToolDependencies |
| `console.py` | Fork audio to SpeakerManager |
| `openai_realtime.py` | Inject speaker context |
| `profiles/main/tools.txt` | Add `enroll_voice` |
| `profiles/main/instructions.txt` | Add enrollment instructions |

## Data Flow: Identification

```
1. User speaks
2. Audio frames accumulated (~3 seconds)
3. SpeakerManager.process(audio_buffer)
4. pyannote extracts 512-dim embedding
5. Compare to enrolled embeddings (cosine similarity)
6. Best match "eric" with similarity 0.82 > 0.75 threshold
7. SpeakerManager.current_speaker = "eric"
8. ToolDependencies.current_user_id updated
9. Next response uses Eric's context/memories
```

## Data Flow: Enrollment

```
1. User: "Reachy, remember my voice, I'm Sarah"
2. LLM calls enroll_voice(name="sarah")
3. SpeakerManager enters enrollment mode
4. Reachy: "Sure Sarah, keep talking for a few seconds..."
5. Audio accumulated (~5 seconds for better quality)
6. pyannote extracts embedding
7. Save to ~/.reachy_mini/voiceprints/sarah.npy
8. Add to active speaker list
9. Reachy: "Got it! I'll recognize your voice now."
```

## Voiceprint Storage Format

```
~/.reachy_mini/
└── voiceprints/
    ├── manifest.json      # {"eric": "eric.npy", "sarah": "sarah.npy"}
    ├── eric.npy           # numpy array (512,)
    └── sarah.npy          # numpy array (512,)
```

## Performance Notes

- pyannote embedding: ~57ms per inference
- Apple MPS acceleration available
- Model downloads on first use (~15MB)
- Embeddings are small (512 floats = 2KB per speaker)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| No HF token | Graceful fallback to config-based user ID |
| Model download slow | Cache model, show progress |
| Similar voices | Confidence threshold + ask to confirm |
| Audio format mismatch | Resample to 16kHz mono |

## Testing Plan

1. **Unit tests**: Mock pyannote, test SpeakerManager logic
2. **Integration**: Enroll test voice, verify identification
3. **Latency**: Benchmark on Mac to confirm <100ms
