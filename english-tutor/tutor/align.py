"""Phonetically weighted alignment of expected vs. heard phonemes."""
from __future__ import annotations

from dataclasses import dataclass

from .phonemes import is_vowel

# Pairs that are close (cheap to substitute) — voicing pairs, near vowels.
_CLOSE = {
    frozenset(p)
    for p in [
        ("t", "d"), ("p", "b"), ("k", "g"), ("s", "z"), ("f", "v"), ("θ", "ð"),
        ("ʃ", "ʒ"), ("tʃ", "dʒ"), ("t", "ɾ"), ("d", "ɾ"),
        ("i", "ɪ"), ("u", "ʊ"), ("ɛ", "æ"), ("ə", "ʌ"), ("ɑ", "ɔ"), ("ɑ", "ɒ"),
        ("ə", "ɪ"), ("ɜ", "ɚ"), ("ə", "ɚ"), ("ɛ", "eɪ"), ("ɔ", "oʊ"),
    ]
}
INS_DEL = 1.0


def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if frozenset((a, b)) in _CLOSE:
        return 0.4
    va, vb = is_vowel(a), is_vowel(b)
    if va and vb:
        return 0.7
    if not va and not vb:
        return 0.9
    return 1.4


@dataclass
class Op:
    kind: str  # "match" | "sub" | "ins" | "del"
    exp: str | None
    heard: str | None
    exp_idx: int | None  # index into the expected sequence (for ins: index of the NEXT expected phone)


def align(expected: list[str], heard: list[str]) -> tuple[list[Op], float]:
    n, m = len(expected), len(heard)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0], B[i][0] = i * INS_DEL, "del"
    for j in range(1, m + 1):
        D[0][j], B[0][j] = j * INS_DEL, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = sub_cost(expected[i - 1], heard[j - 1])
            options = (
                (D[i - 1][j - 1] + c, "diag"),
                (D[i - 1][j] + INS_DEL, "del"),
                (D[i][j - 1] + INS_DEL, "ins"),
            )
            D[i][j], B[i][j] = min(options, key=lambda o: o[0])
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i][j]
        if b == "diag":
            e, h = expected[i - 1], heard[j - 1]
            ops.append(Op("match" if e == h else "sub", e, h, i - 1))
            i, j = i - 1, j - 1
        elif b == "del":
            ops.append(Op("del", expected[i - 1], None, i - 1))
            i -= 1
        else:
            ops.append(Op("ins", None, heard[j - 1], i))
            j -= 1
    ops.reverse()
    return ops, D[n][m]
