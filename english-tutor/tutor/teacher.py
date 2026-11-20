"""The teacher brain: Claude + method + program + what we know about the student."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

from .pronunciation import Report

MODEL = "claude-opus-5-5"
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "students"


def _load_curriculum() -> dict:
    return json.loads((HERE / "method" / "curriculum.json").read_text())


@dataclass
class Student:
    id: str
    name: str = ""
    level: str = "A2"
    unit_id: str = "u1"
    issue_counts: dict[str, int] = field(default_factory=dict)
    issue_words: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, sid: str) -> "Student":
        p = DATA / f"{sid}.json"
        return cls(**json.loads(p.read_text())) if p.exists() else cls(id=sid)

    def save(self) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / f"{self.id}.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    def record(self, report: Report) -> None:
        for i in report.issues:
            self.issue_counts[i.type] = self.issue_counts.get(i.type, 0) + 1
            words = self.issue_words.setdefault(i.type, [])
            if i.word.lower() not in words:
                words[:] = (words + [i.word.lower()])[-8:]

    def summary(self) -> str:
        top = sorted(self.issue_counts.items(), key=lambda kv: -kv[1])[:5]
        recurring = "; ".join(f"{t} x{n} (e.g. {', '.join(self.issue_words.get(t, [])[-3:])})" for t, n in top)
        return (
            f"Name: {self.name or 'unknown (ask naturally)'}\nLevel: {self.level}\n"
            f"Current unit: {self.unit_id}\nRecurring pronunciation issues: {recurring or 'none recorded yet'}"
        )


@dataclass
class Turn:
    transcript: str
    report: Report | None = None

    def to_message(self) -> str:
        parts = [f"<student_said>{self.transcript}</student_said>"]
        if self.report is not None:
            parts.append(
                "<pronunciation_report>"
                + json.dumps(self.report.for_teacher(), ensure_ascii=False)
                + "</pronunciation_report>"
            )
        return "\n".join(parts)


class Teacher:
    def __init__(self, student: Student, client: anthropic.Anthropic | None = None, effort: str = "low"):
        self.client = client or anthropic.Anthropic()
        self.student = student
        self.effort = effort  # low = fast, conversational replies
        self.history: list[dict] = []
        method = (HERE / "method" / "base_method.md").read_text()
        self._stable_system = (
            method + "\n\n## PROGRAM (units in order)\n" + json.dumps(_load_curriculum(), ensure_ascii=False)
        )

    def _system(self) -> list[dict]:
        return [
            {"type": "text", "text": self._stable_system, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "## THIS STUDENT\n" + self.student.summary()},
        ]

    def reply_stream(self, turn: Turn) -> Iterator[str]:
        """Yields the teacher's reply as it is generated (feed sentences to TTS as they arrive)."""
        if turn.report is not None:
            self.student.record(turn.report)
            self.student.save()
        self.history.append({"role": "user", "content": turn.to_message()})
        with self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=2000,
            system=self._system(),
            messages=self.history,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            text = "Sorry, let's talk about something else. What did you do today?"
            self.history.append({"role": "assistant", "content": text})
            yield text
            return
        self.history.append({"role": "assistant", "content": final.content})

    def reply(self, turn: Turn) -> str:
        return "".join(self.reply_stream(turn))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentences(chunks: Iterator[str]) -> Iterator[str]:
    """Regroup a token stream into whole sentences, so speech can start before the reply is finished."""
    buf = ""
    for c in chunks:
        buf += c
        parts = _SENTENCE_END.split(buf)
        for s in parts[:-1]:
            if s.strip():
                yield s.strip()
        buf = parts[-1]
    if buf.strip():
        yield buf.strip()
