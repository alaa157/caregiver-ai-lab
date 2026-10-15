"""Tests for Arabic normalization helpers."""
from src.normalization import normalize_arabic, normalize, split_sentences, word_tokens, content_words


def test_alef_variants_normalize_to_plain_alef():
    for variant in ["إبراهيم", "أحمد", "آية"]:
        result = normalize_arabic(variant)
        assert "إ" not in result and "أ" not in result and "آ" not in result
    assert normalize_arabic("إبراهيم") == "ابراهيم"
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("آية") == "اية".replace("ة", "ه")


def test_tatweel_removed():
    assert normalize_arabic("شـــــكرا") == "شكرا"


def test_harakat_removed():
    assert normalize_arabic("عَرَبِيّ") == "عربي"


def test_arabic_punctuation_mapped():
    assert normalize_arabic("نعم، شكرًا؛ alright؟") == "نعم, شكرا; alright?"


def test_normalize_lowercases():
    assert normalize("SALAM") == "salam"
    assert normalize("سَلام") == "سلام"


def test_split_sentences_arabic_boundary():
    parts = split_sentences("أكلت جيداً. لم تنم؟")
    assert len(parts) == 2
    assert parts[0] == "أكلت جيداً."


def test_content_words_filters_short():
    assert content_words("هو أكل") == ["اكل"]
