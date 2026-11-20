"""Command line entry point.

  python -m tutor check "I worked yesterday" --said-like "I workid yesterday"   # simulate a mistake
  python -m tutor check "I worked yesterday" --audio me.wav                     # real recording
  python -m tutor listen me.wav                                                 # transcribe + assess free speech
  python -m tutor speak "Hello, how are you?" -o hello.wav
  python -m tutor chat --student rafael                                         # text chat with the teacher
"""
from __future__ import annotations

import argparse
import json

from .phonemes import text_to_phones
from .pronunciation import assess


def _print_report(report) -> None:
    print(f"Overall: {report.overall_score:.0%}  (closest accent: {report.accent_matched})")
    for w in report.words:
        mark = "OK " if not w.issues else "!! "
        print(f"  {mark}{w.word:<14} expected [{w.expected}]  heard [{w.heard}]")
        for i in w.issues:
            print(f"       -> {i.type} ({i.severity}): {i.tip_pt}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="tutor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="assess pronunciation of a known sentence")
    c.add_argument("text")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio")
    g.add_argument("--said-like", help="simulate: phonemise this respelling as if it were heard")
    c.add_argument("--json", action="store_true")

    l = sub.add_parser("listen", help="free speech: transcribe with Whisper, then assess")
    l.add_argument("audio")

    s = sub.add_parser("speak", help="teacher voice to a wav file")
    s.add_argument("text")
    s.add_argument("-o", "--out", default="teacher.wav")

    ch = sub.add_parser("chat", help="text conversation with the teacher (type 'quit' to stop)")
    ch.add_argument("--student", default="demo")
    ch.add_argument("--said-like", action="store_true",
                    help="write 'what you meant || how you said it' to simulate pronunciation")

    a = ap.parse_args()

    if a.cmd == "check":
        if a.audio:
            from .speech import PhonemeRecognizer, load_audio

            heard = PhonemeRecognizer().recognize(load_audio(a.audio))
        else:
            heard = text_to_phones(a.said_like)
        r = assess(a.text, heard)
        print(json.dumps(r.for_teacher(), ensure_ascii=False, indent=2) if a.json else "", end="")
        if not a.json:
            _print_report(r)

    elif a.cmd == "listen":
        from .speech import PhonemeRecognizer, Transcriber, load_audio

        audio = load_audio(a.audio)
        text = Transcriber().transcribe(audio)
        print(f"Transcript: {text}")
        _print_report(assess(text, PhonemeRecognizer().recognize(audio)))

    elif a.cmd == "speak":
        from .speech import Voice

        print(Voice().to_wav(a.text, a.out))

    elif a.cmd == "chat":
        from .teacher import Student, Teacher, Turn

        t = Teacher(Student.load(a.student))
        print("Teacher is ready. Say hi! ('quit' to stop)")
        while (line := input("\nyou> ").strip()) not in ("quit", "exit"):
            report = None
            if a.said_like and "||" in line:
                meant, said = (x.strip() for x in line.split("||", 1))
                report, line = assess(meant, text_to_phones(said)), meant
            print("teacher> ", end="", flush=True)
            for chunk in t.reply_stream(Turn(line, report)):
                print(chunk, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
