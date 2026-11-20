"""Pronunciation assessment: compare what the student SHOULD have said with
the phonemes they ACTUALLY produced, word by word."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .align import align
from .l1_pt import Issue, WordOp, detect
from .phonemes import WEAK_FORMS, is_vowel, normalise_seq, reference

ACCENTS = ("en-us", "en-gb")  # either accent counts as correct


@dataclass
class WordResult:
    word: str
    expected: str
    heard: str
    score: float
    issues: list[Issue] = field(default_factory=list)


@dataclass
class Report:
    text: str
    accent_matched: str
    overall_score: float
    words: list[WordResult]

    @property
    def issues(self) -> list[Issue]:
        return [i for w in self.words for i in w.issues]

    def for_teacher(self) -> dict:
        """Compact version handed to the teacher brain."""
        return {
            "overall_score": round(self.overall_score, 2),
            "words_with_issues": [
                {
                    "word": w.word,
                    "expected_ipa": w.expected,
                    "heard_ipa": w.heard,
                    "issues": [
                        {"type": i.type, "severity": i.severity, "tip_pt": i.tip_pt, "tip_en": i.tip_en}
                        for i in w.issues
                    ],
                }
                for w in self.words
                if w.issues
            ],
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _assess_one(text: str, heard: list[str], accent: str) -> tuple[Report, float]:
    ref = reference(text, accent)
    flat: list[str] = []
    owner: list[int] = []  # expected phone index -> word index
    starts: list[int] = []
    for wi, w in enumerate(ref):
        starts.append(len(flat))
        flat.extend(w.phones)
        owner.extend([wi] * len(w.phones))
    ops, cost = align(flat, heard)

    per_word: list[list[WordOp]] = [[] for _ in ref]
    for o in ops:
        if o.kind != "ins":
            wi = owner[o.exp_idx]
            per_word[wi].append(WordOp(o.kind, o.exp, o.heard, o.exp_idx - starts[wi]))
            continue
        nxt = o.exp_idx  # insertion sits before expected phone `nxt`
        if nxt >= len(flat):
            wi = len(ref) - 1
        elif nxt in starts:  # word boundary: belongs to previous word unless it is s-cluster epenthesis
            wi_next = owner[nxt]
            ph = ref[wi_next].phones
            s_cluster = len(ph) > 1 and ph[0] == "s" and not is_vowel(ph[1]) and is_vowel(o.heard)
            wi = wi_next if (s_cluster or wi_next == 0) else wi_next - 1
        else:
            wi = owner[nxt]
        pos = nxt - starts[wi] if wi < len(ref) else 0
        per_word[wi].append(WordOp("ins", None, o.heard, max(0, min(pos, len(ref[wi].phones)))))

    words: list[WordResult] = []
    for w, wops in zip(ref, per_word):
        n = max(1, len(w.phones))
        from .align import sub_cost

        wcost = sum(1.0 if o.kind in ("ins", "del") else sub_cost(o.exp, o.heard) for o in wops if o.kind != "match")
        heard_w = [o.heard for o in wops if o.heard]
        if heard_w in WEAK_FORMS.get(w.text.lower(), []):
            wcost, issues = 0.0, []  # natural weak/strong form: correct
        else:
            issues = detect(w.text, w.phones, wops)
        words.append(
            WordResult(
                word=w.text,
                expected=" ".join(w.phones),
                heard=" ".join(heard_w),
                score=max(0.0, 1 - wcost / n),
                issues=issues,
            )
        )
    overall = sum(w.score for w in words) / max(1, len(words))
    return Report(text, accent, overall, words), cost


def assess(text: str, heard_phones: list[str], accents=ACCENTS) -> Report:
    """`text` = what the student meant to say; `heard_phones` = recogniser output."""
    heard = normalise_seq(heard_phones)
    best = min((_assess_one(text, heard, a) for a in accents), key=lambda rc: rc[1])
    return best[0]
