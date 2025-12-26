# Speaker Identification v2

**Status**: Implemented
**Goal**: Simplify speaker identification architecture
**Last Updated**: 2024-12-26

---

## Summary

Simplified speaker identification by:
- **Keeping** real-time identification (6s embedding comparison)
- **Removing** session tracker (~300 lines of complex clustering code)
- **Removing** immediate enrollment (5s was noisy)
- **Using** diarization for ALL enrollment (cleaner audio, better quality)
- **Reducing** buffer size from 15 min to 5 min

## Architecture

```
Audio → embedding (6s) → compare to enrolled → identify speaker
              ↓ if unknown
       trigger background diarization
              ↓
       diarization matches speakers to names/voiceprints
              ↓
       enroll if speaker has ≥20s of speech
```

## Key Changes Made

### Removed
- `speaker/session_tracker.py` - entire file deleted
- `SessionSpeaker` dataclass
- `SessionSpeakerTracker` class
- `get_unenrolled_speakers_with_names()` method
- `assign_name_to_speaker()` method
- `enroll_session_speaker()` method
- `start_enrollment()` method - old immediate enrollment
- `_complete_enrollment()` method - old immediate enrollment
- `cancel_enrollment()` method - old immediate enrollment
- `is_enrolling` property
- `AUDIO_DURATION_ENROLL` constant

### Added
- `SPEAKER_MIN_ENROLLMENT_SECONDS` config (20s default)
- `_enroll_from_diarization()` method in SpeakerManager
- Updated `_on_diarization_results()` to actually enroll speakers
- Tentative speaker identification on name mention

### Modified
- `config.py` - new enrollment config, reduced buffer to 300s
- `speaker_manager.py` - removed session tracker & immediate enrollment, simplified
- `openai_realtime.py` - simplified conversation end, removed session enrollment
- `main.py` - pass mentioned_names to SpeakerManager
- `tools/enroll_voice.py` - now uses diarization-based enrollment
- `profiles/main/instructions.txt` - updated enrollment instructions (20-30s)

## Flow

### Known Speaker (Already Enrolled)
```
Eric speaks → 6s embedding → match eric.npy (0.72) → "Eric"
→ No diarization needed
```

### Unknown Speaker, Introduces Themselves
```
Alex speaks → 6s embedding → no match → unknown → triggers diarization
Alex: "I'm Alex" → store in mentioned_names, set tentative current_user_id

~60s later, diarization completes:
→ SPEAKER_01 matched to "alex" via timing
→ SPEAKER_01 has 25s speech (≥20s minimum)
→ Enroll alex.npy from clean segments

Next 6s cycle:
→ Match alex.npy → "Alex" (confirmed)
```

### Explicit Enrollment via Tool
```
User: "Remember my voice, I'm Sarah"
→ enroll_voice tool called with name="sarah"
→ Name stored in mentioned_names, diarization triggered
→ User keeps talking naturally for ~20-30s
→ Diarization completes, enrolls sarah.npy from clean segments
```

## Files Changed

| File | Change |
|------|--------|
| `speaker/session_tracker.py` | **DELETED** |
| `speaker/speaker_manager.py` | Removed session tracker & immediate enrollment, added diarization enrollment |
| `config.py` | Added MIN_ENROLLMENT_SECONDS, reduced buffer |
| `openai_realtime.py` | Simplified conversation end |
| `main.py` | Pass mentioned_names to SpeakerManager |
| `tools/enroll_voice.py` | Diarization-based enrollment |
| `profiles/main/instructions.txt` | Updated enrollment time (20-30s) |
