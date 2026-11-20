# English Tutor (voice, for Brazilian Portuguese speakers)

A private English teacher you talk to. It hears **how** you pronounce things, not just what you
say, and corrects you inside a teaching method.

## How a turn works

```
student audio ─┬─► Whisper (open source) ──────────► what was said
               └─► wav2vec2 phoneme model (open) ───► the sounds actually made
                              │
                 pronunciation engine (this repo): expected vs. heard, sound by sound,
                 + Brazilian-learner error patterns (-ed as extra syllable, "estudy",
                 "bigi", th → t, r → h, ship/sheep, cat/bed, final l → u ...)
                              │
                 Claude (teacher brain): method + program + student history
                              │
                 Kokoro voice (open source) ─► spoken reply
```

## Status

| Piece | State |
|---|---|
| Pronunciation engine (alignment + Portuguese-speaker patterns) | ✅ built, 13 tests passing (simulated input) |
| Teacher brain (base method, 5-unit program, student memory) | ✅ built, needs `ANTHROPIC_API_KEY` to run |
| Teacher voice (Kokoro) | ✅ working (~3 s per sentence on a 4-core CPU) |
| Phoneme recogniser + Whisper | ✅ code written, **not yet tested on real audio** (models download from Hugging Face) |
| Owner's method | ⏳ placeholder in `tutor/method/base_method.md` |
| Web app (phone + computer, push-to-talk) | ⏳ next |

## Setup

```bash
sudo apt install espeak-ng ffmpeg
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # only for the teacher brain
```

## Try it

```bash
# Pronunciation, simulating a mistake (no audio needed)
python -m tutor check "I worked yesterday" --said-like "I workid yesterday"

# Pronunciation from a real recording of a known sentence
python -m tutor check "I worked yesterday" --audio me.wav

# Free speech: transcribe, then assess
python -m tutor listen me.wav

# Teacher voice
python -m tutor speak "Say it with me: worked." -o out.wav

# Text chat with the teacher; "meant || how you said it" simulates pronunciation
python -m tutor chat --student rafael --said-like
#   you> I worked a lot yesterday || I workid a lot yesterday

pytest -q
```

## Where things live

- `tutor/phonemes.py`: expected pronunciation (espeak-ng), normalisation, weak forms
- `tutor/align.py`: sound-by-sound alignment with phonetic similarity
- `tutor/l1_pt.py`: Brazilian Portuguese speaker error patterns, with tips in PT and EN
- `tutor/pronunciation.py`: the report handed to the teacher
- `tutor/speech.py`: open-source recogniser, Whisper and Kokoro voice
- `tutor/teacher.py`: Claude teacher, student profile, sentence streaming for low-latency speech
- `tutor/method/`: teaching method and program (edit these to change how it teaches)
