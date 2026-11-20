"""Reference pronunciations and phoneme normalisation.

Reference phonemes come from espeak-ng (via `phonemizer`). The open-source
recogniser we use (wav2vec2 "espeak" phoneme model) was trained on the same
espeak IPA inventory, so both sides speak the same phoneme "alphabet".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

VOWELS = set("aeiouæɑɒɐəɚɛɜɝɪɨʊʌɔøœɤɯyᵻ")

# Composite tokens espeak emits that we split so both sides are comparable.
_SPLITS = {
    "əl": ["ə", "l"],
    "ɑɹ": ["ɑ", "ɹ"],
    "ɔɹ": ["ɔ", "ɹ"],
    "ɛɹ": ["ɛ", "ɹ"],
    "ɪɹ": ["ɪ", "ɹ"],
    "ʊɹ": ["ʊ", "ɹ"],
    "aɪɚ": ["aɪ", "ɚ"],
    "aʊɚ": ["aʊ", "ɚ"],
}

# Symbol-level equivalences (variants that should not count as errors).
_EQUIV = {
    "ɐ": "ə",
    "ᵻ": "ɪ",
    "ɝ": "ɜ",
    "r": "ɹ",
    "ɡ": "g",
    "e": "eɪ",  # some models emit bare e for the FACE vowel
    "o": "oʊ",
    "əʊ": "oʊ",  # British GOAT
}


def normalise(token: str) -> list[str]:
    """Strip stress/length marks and split composite tokens."""
    t = re.sub(r"[ˈˌː̯͡]", "", token.strip())
    if not t:
        return []
    if t in _SPLITS:
        return [_EQUIV.get(p, p) for p in _SPLITS[t]]
    return [_EQUIV.get(t, t)]


def normalise_seq(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        out.extend(normalise(tok))
    return out


def is_vowel(p: str) -> bool:
    return bool(p) and p[0] in VOWELS


# Function words have relaxed "weak" forms in natural speech. Any of these is correct.
WEAK_FORMS: dict[str, list[list[str]]] = {
    "a": [["ə"], ["eɪ"]],
    "an": [["ə", "n"], ["æ", "n"]],
    "the": [["ð", "ə"], ["ð", "i"], ["ð", "ɪ"]],
    "to": [["t", "ə"], ["t", "u"], ["t", "ʊ"], ["ɾ", "ə"]],
    "and": [["ə", "n", "d"], ["ə", "n"], ["æ", "n", "d"], ["n"]],
    "of": [["ə", "v"], ["ʌ", "v"], ["ə"]],
    "for": [["f", "ɚ"], ["f", "ɔ", "ɹ"], ["f", "ə"]],
    "was": [["w", "ə", "z"], ["w", "ʌ", "z"], ["w", "ɑ", "z"]],
    "can": [["k", "ə", "n"], ["k", "æ", "n"]],
    "at": [["ə", "t"], ["æ", "t"]],
    "are": [["ɚ"], ["ɑ", "ɹ"]],
    "you": [["j", "ə"], ["j", "u"]],
    "your": [["j", "ɚ"], ["j", "ɔ", "ɹ"], ["j", "ʊ", "ɹ"]],
    "some": [["s", "ə", "m"], ["s", "ʌ", "m"]],
    "but": [["b", "ə", "t"], ["b", "ʌ", "t"]],
    "from": [["f", "ɹ", "ə", "m"], ["f", "ɹ", "ʌ", "m"]],
}


@dataclass
class WordRef:
    text: str
    phones: list[str]


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words_of(text: str) -> list[str]:
    return _WORD_RE.findall(text)


@lru_cache(maxsize=4)
def _backend(lang: str):
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(lang, with_stress=False, preserve_punctuation=False)


@lru_cache(maxsize=4096)
def word_phones(word: str, lang: str = "en-us") -> tuple[str, ...]:
    from phonemizer.separator import Separator

    out = _backend(lang).phonemize([word], separator=Separator(phone=" ", word="|"), strip=True)[0]
    toks = [t for t in out.replace("|", " ").split(" ") if t]
    return tuple(normalise_seq(toks))


@lru_cache(maxsize=1024)
def _sentence_phones(text: str, lang: str) -> tuple[tuple[str, ...], ...]:
    from phonemizer.separator import Separator

    out = _backend(lang).phonemize([text], separator=Separator(phone=" ", word="|"), strip=True)[0]
    return tuple(tuple(normalise_seq([t for t in w.split(" ") if t])) for w in out.split("|") if w.strip())


def reference(text: str, lang: str = "en-us") -> list[WordRef]:
    """Expected pronunciation, word by word.

    The whole sentence is phonemised at once so function words get their natural
    weak forms ("a" = ə, not "ay"). If espeak splits words differently from us,
    fall back to word-by-word.
    """
    words = words_of(text)
    in_context = _sentence_phones(" ".join(words).lower(), lang)
    if len(in_context) == len(words):
        return [WordRef(w, list(p)) for w, p in zip(words, in_context)]
    return [WordRef(w, list(word_phones(w.lower(), lang))) for w in words]


def text_to_phones(text: str, lang: str = "en-us") -> list[str]:
    """Flat phoneme string for a piece of text (used to simulate what was heard)."""
    out: list[str] = []
    for w in reference(text, lang):
        out.extend(w.phones)
    return out
