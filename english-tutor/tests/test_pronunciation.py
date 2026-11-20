"""Pronunciation-engine tests.

The recogniser model is not used here: we SIMULATE what was heard by
phonemising a respelling of the mistake (e.g. "workid" for "worked").
That tests alignment + Portuguese-speaker error detection in isolation.
"""
import pytest

from tutor.phonemes import text_to_phones
from tutor.pronunciation import assess


def issue_types(target: str, said_like: str) -> set[str]:
    r = assess(target, text_to_phones(said_like))
    return {i.type for i in r.issues}


def test_correct_pronunciation_has_no_issues():
    r = assess("I worked yesterday", text_to_phones("I worked yesterday"))
    assert r.issues == []
    assert r.overall_score == pytest.approx(1.0)


def test_british_accent_is_accepted():
    from tutor.phonemes import reference

    gb = [p for w in reference("I worked on the car", "en-gb") for p in w.phones]
    assert assess("I worked on the car", gb).issues == []


@pytest.mark.parametrize(
    "target,said_like,expected",
    [
        ("I worked yesterday", "I workid yesterday", "ed_ending"),
        ("She liked it", "She likid it", "ed_ending"),
        ("I study English", "I estudy English", "s_cluster_vowel"),
        ("a big dog", "a bigi dog", "final_vowel_added"),
        ("I think so", "I tink so", "th_sound"),
        ("the red car", "the head car", "r_as_h"),
        ("a big ship", "a big sheep", "i_vs_ɪ"),
        ("a bad day", "a bed day", "ae_vowel"),
    ],
)
def test_portuguese_speaker_patterns(target, said_like, expected):
    assert expected in issue_types(target, said_like)


def test_wanted_keeps_its_extra_syllable():
    # "wanted" really has /ɪd/ — saying it that way must NOT be flagged.
    assert "ed_ending" not in issue_types("I wanted it", "I wanted it")


def test_issue_points_to_the_right_word():
    r = assess("I worked yesterday", text_to_phones("I workid yesterday"))
    flagged = [w.word for w in r.words if w.issues]
    assert flagged == ["worked"]


def test_weak_forms_of_function_words_are_accepted():
    heard = ["ɪ", "t", "w", "ə", "z", "ə", "b", "ɪ", "g", "d", "eɪ", "f", "ɚ", "ð", "ə", "t", "i", "m"]
    assert assess("It was a big day for the team", heard).issues == []
