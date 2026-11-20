"""Typical pronunciation patterns of Brazilian Portuguese speakers learning English.

Each detector looks at how one word was aligned (expected vs. heard phonemes)
and turns raw differences into a named, teachable issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .phonemes import is_vowel

EPENTHETIC = {"ɪ", "i", "ə", "ɨ", "ɛ", "eɪ", "e", "ʊ", "u"}

# type -> (severity, tip in English, tip in Portuguese)
CATALOG: dict[str, tuple[str, str, str]] = {
    "ed_ending": (
        "high",
        "The -ed ending is not an extra syllable here. After this sound it is just /t/ or /d/ glued to the word (worked = 'workt').",
        "O -ed aqui não vira sílaba extra. Depois desse som ele é só /t/ ou /d/ colado na palavra (worked = 'uôrkt', e não 'uôrkid').",
    ),
    "final_vowel_added": (
        "medium",
        "An extra vowel was added after the final consonant. End the word on the consonant.",
        "Você colocou uma vogal extra depois da consoante final (tipo 'bigui'). Termine a palavra na consoante.",
    ),
    "s_cluster_vowel": (
        "medium",
        "An extra vowel was added before the initial s. Start directly with the hissing 'sss'.",
        "Você colocou um 'i/e' antes do s inicial ('estudy'). Comece direto no 'sss': study.",
    ),
    "th_sound": (
        "medium",
        "The 'th' was replaced. Put the tip of the tongue between the teeth and blow air.",
        "O 'th' saiu como outro som. Coloque a ponta da língua entre os dentes e solte o ar.",
    ),
    "r_as_h": (
        "high",
        "The English r sounded like the Portuguese 'rr' (an h sound). Curl the tongue back without touching the roof of the mouth.",
        "O r saiu como o 'rr' do português (som de h), e 'red' vira 'head'. Enrole a língua para trás, sem encostar no céu da boca.",
    ),
    "dark_l": (
        "low",
        "The final l turned into a 'u' sound. Touch the tongue tip behind the top teeth at the end.",
        "O l final virou 'u' (como em 'Brasil'). No inglês, encoste a ponta da língua atrás dos dentes de cima no final.",
    ),
    "i_vs_ɪ": (
        "high",
        "Long 'ee' (sheep) and short 'i' (ship) got mixed up. This changes the meaning of words.",
        "Misturou o 'i' longo (sheep) com o 'i' curto e relaxado (ship). Isso muda o significado da palavra.",
    ),
    "u_vs_ʊ": (
        "medium",
        "Long 'oo' (food) and short 'u' (foot) got mixed up.",
        "Misturou o 'u' longo (food) com o 'u' curto (foot).",
    ),
    "ae_vowel": (
        "medium",
        "The 'a' in words like cat/bad needs the mouth wide open, between 'é' and 'á'.",
        "O 'a' de cat/bad é um som que não existe em português: boca bem aberta, entre 'é' e 'á'.",
    ),
    "t_d_palatalized": (
        "low",
        "t/d before an 'i' sound turned into 'tch/dj'. Keep it a clean t or d.",
        "O t/d antes de 'i' virou 'tchi/dji' (como no português do Brasil). Mantenha t ou d limpo.",
    ),
    "final_nasal": (
        "medium",
        "The final m/n/ng was swallowed into a nasal vowel. Close the sound clearly at the end.",
        "O m/n final sumiu e virou só vogal nasal. Feche o som no final: 'tiME', 'fuN'.",
    ),
    "final_consonant_dropped": (
        "high",
        "The final consonant was dropped. Pronounce the last sound of the word.",
        "A consoante final sumiu. Pronuncie o último som da palavra.",
    ),
    "h_dropped": (
        "medium",
        "The initial h was not pronounced. Breathe out a soft 'h' (like the Portuguese 'rr' in 'rato', but softer).",
        "O h inicial não foi pronunciado. Solte um 'h' suave de ar (parecido com o 'r' de 'rato', bem leve).",
    ),
    "other": (
        "medium",
        "This sound was different from the expected pronunciation.",
        "Esse som saiu diferente da pronúncia esperada.",
    ),
}


@dataclass
class Issue:
    word: str
    type: str
    expected: str
    heard: str
    severity: str = ""
    tip_en: str = ""
    tip_pt: str = ""

    def __post_init__(self):
        sev, en, pt = CATALOG[self.type]
        self.severity = self.severity or sev
        self.tip_en = self.tip_en or en
        self.tip_pt = self.tip_pt or pt


@dataclass
class WordOp:
    kind: str
    exp: str | None
    heard: str | None
    pos: int  # position in the word's expected phones (for "ins": index of the next phone)


@dataclass
class _Ctx:
    word: str
    phones: list[str]
    ops: list[WordOp]
    used: set[int] = field(default_factory=set)


def detect(word: str, phones: list[str], ops: list[WordOp]) -> list[Issue]:
    ctx = _Ctx(word, phones, ops)
    issues: list[Issue] = []
    exp_s = " ".join(phones)
    heard_s = " ".join(o.heard for o in ops if o.heard)
    n = len(phones)
    w = word.lower()

    def add(t: str, idxs: list[int], exp=exp_s, heard=heard_s):
        ctx.used.update(idxs)
        issues.append(Issue(word, t, exp, heard))

    # --- insertions -------------------------------------------------------
    ed_word = w.endswith("ed") and n >= 2 and phones[-1] in ("t", "d") and phones[-2] not in ("t", "d")
    for k, o in enumerate(ops):
        if o.kind != "ins" or not o.heard or not is_vowel(o.heard):
            continue
        if ed_word and o.pos >= n - 1:
            add("ed_ending", [k])
        elif o.pos == 0 and n > 1 and phones[0] == "s" and not is_vowel(phones[1]):
            add("s_cluster_vowel", [k])
        elif o.pos == n and n and not is_vowel(phones[-1]) and o.heard in EPENTHETIC:
            add("final_vowel_added", [k])

    # --- substitutions / deletions ----------------------------------------
    for k, o in enumerate(ops):
        if k in ctx.used or o.kind not in ("sub", "del"):
            continue
        e, h, last = o.exp, o.heard, o.pos == n - 1
        nxt = phones[o.pos + 1] if o.pos + 1 < n else ""
        if o.kind == "sub":
            if (e == "θ" and h in ("t", "f", "s")) or (e == "ð" and h in ("d", "z", "v")):
                add("th_sound", [k])
            elif e == "ɹ" and h in ("h", "x", "χ", "ʁ"):
                add("r_as_h", [k])
            elif e == "l" and last and h in ("ʊ", "u", "w", "oʊ"):
                add("dark_l", [k])
            elif {e, h} == {"i", "ɪ"}:
                add("i_vs_ɪ", [k])
            elif {e, h} == {"u", "ʊ"}:
                add("u_vs_ʊ", [k])
            elif e == "æ" and h in ("ɛ", "eɪ", "a", "ɑ"):
                add("ae_vowel", [k])
            elif (e, h) in (("t", "tʃ"), ("d", "dʒ")) and nxt in ("i", "ɪ"):
                add("t_d_palatalized", [k])
            elif ed_word and last and (e, h) in (("t", "d"), ("d", "t")):
                ctx.used.add(k)  # voicing of -ed is harmless
        else:  # deletion
            if last and e in ("m", "n", "ŋ"):
                add("final_nasal", [k])
            elif last and not is_vowel(e):
                add("final_consonant_dropped", [k])
            elif o.pos == 0 and e == "h":
                add("h_dropped", [k])

    # --- anything left unexplained ----------------------------------------
    leftovers = [k for k, o in enumerate(ops) if o.kind != "match" and k not in ctx.used]
    # Ignore near-identical substitutions (accent variation, not errors).
    from .align import sub_cost

    leftovers = [k for k in leftovers if not (ops[k].kind == "sub" and sub_cost(ops[k].exp, ops[k].heard) < 0.5)]
    if leftovers:
        add("other", leftovers)
    return issues
