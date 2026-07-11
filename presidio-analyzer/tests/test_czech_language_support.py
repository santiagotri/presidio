"""
End-to-end tests for Czech (cs) language support.

Reproduces the two historical failure modes and verifies both are fixed:

(a) Standard entities (PERSON, LOCATION) were not detected because no NER
    model was configured for Czech, and DATE_TIME was not detected because
    no recognizer covered Czech date notation. NER output is simulated
    here through precomputed ``NlpArtifacts`` (labels already mapped to
    Presidio entities, exactly as the NLP engine configured in
    ``docs/recipes/czech-language-support/spacy_en_cs.yaml`` produces
    them), so the test runs without downloading a model. DATE_TIME comes
    from the real ``CzDateRecognizer``.

(b) Czech identifiers collided with PHONE_NUMBER: the phone recognizer
    used to win on ``850412/0003`` (rodné číslo) and on part of
    ``19-123456789/0800`` (bank account). The Czech recognizers use
    context words and validation to outscore PHONE_NUMBER so that
    score-based conflict resolution (as applied by presidio-anonymizer)
    picks the Czech entity.

All personal data in the test text is fictitious.
"""
from typing import Dict, List, Tuple

import pytest
import spacy

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers import (
    CzBankAccountRecognizer,
    CzBirthNumberRecognizer,
    CzDateRecognizer,
    CzDriverLicenseRecognizer,
    CzIdCardRecognizer,
    CzPassportRecognizer,
    DateRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)
from tests.mocks import NlpEngineMock

TEXT = (
    "Jmenuji se Jan Novák a narodil jsem se dne 12. dubna 1985. "
    "Moje rodné číslo je 850412/0003. "
    "Bydlím na adrese Václavské náměstí 123/45, 110 00 Praha 1. "
    "Kontaktovat mě můžete na telefonním čísle +420 601 234 567 "
    "nebo prostřednictvím e-mailové adresy jan.novak@example.cz. "
    "Moje číslo občanského průkazu je 123456789 a číslo cestovního pasu je "
    "AB123456. Řidičský průkaz má číslo CZ123456789. "
    "Bankovní účet mám vedený pod číslem 19-123456789/0800. "
    "IBAN účtu je CZ6508000000192000145399 a BIC/SWIFT kód banky je GIBACZPX."
)

EXPECTED_ANONYMIZED = (
    "Jmenuji se <PERSON_1> a narodil jsem se dne <DATE_TIME_1>. "
    "Moje rodné číslo je <CZ_BIRTH_NUMBER_1>. "
    "Bydlím na adrese <LOCATION_1>. "
    "Kontaktovat mě můžete na telefonním čísle <PHONE_NUMBER_1> "
    "nebo prostřednictvím e-mailové adresy <EMAIL_ADDRESS_1>. "
    "Moje číslo občanského průkazu je <CZ_ID_CARD_1> a číslo cestovního pasu je "
    "<CZ_PASSPORT_1>. Řidičský průkaz má číslo <CZ_DRIVER_LICENSE_1>. "
    "Bankovní účet mám vedený pod číslem <CZ_BANK_ACCOUNT_1>. "
    "IBAN účtu je <IBAN_CODE_1> a BIC/SWIFT kód banky je GIBACZPX."
)

# What the Czech NER model returns for this text, with labels already
# mapped to Presidio entities by the NLP engine (see the ner_model
# configuration in docs/recipes/czech-language-support/spacy_en_cs.yaml).
NER_SPANS = [
    ("Jan Novák", "PERSON"),
    ("Václavské náměstí 123/45, 110 00 Praha 1", "LOCATION"),
]

# Field-by-field expectations from the analyzer (exact span and entity).
EXPECTED_DETECTIONS = [
    ("Jan Novák", "PERSON"),
    ("12. dubna 1985", "DATE_TIME"),
    ("850412/0003", "CZ_BIRTH_NUMBER"),
    ("Václavské náměstí 123/45, 110 00 Praha 1", "LOCATION"),
    ("+420 601 234 567", "PHONE_NUMBER"),
    ("jan.novak@example.cz", "EMAIL_ADDRESS"),
    ("123456789", "CZ_ID_CARD"),
    ("AB123456", "CZ_PASSPORT"),
    ("CZ123456789", "CZ_DRIVER_LICENSE"),
    ("19-123456789/0800", "CZ_BANK_ACCOUNT"),
    ("CZ6508000000192000145399", "IBAN_CODE"),
]


def _czech_nlp_artifacts(text: str) -> NlpArtifacts:
    """Build NlpArtifacts with real Czech tokenization and simulated NER."""
    nlp = spacy.blank("cs")
    doc = nlp(text)
    spans = []
    for span_text, label in NER_SPANS:
        start = text.index(span_text)
        spans.append(
            doc.char_span(
                start, start + len(span_text), label=label, alignment_mode="expand"
            )
        )
    doc.ents = spans
    return NlpArtifacts(
        entities=doc.ents,
        tokens=doc,
        tokens_indices=[token.idx for token in doc],
        lemmas=[token.text for token in doc],
        nlp_engine=NlpEngineMock(),
        language="cs",
    )


def _czech_recognizers() -> List:
    """The recognizer set a Czech pipeline registers for language cs."""
    return [
        CzBirthNumberRecognizer(),
        CzBankAccountRecognizer(),
        CzIdCardRecognizer(),
        CzPassportRecognizer(),
        CzDriverLicenseRecognizer(),
        CzDateRecognizer(),
        # Generic recognizers are instantiated per registry language by
        # the registry provider; mirrored here explicitly.
        PhoneRecognizer(supported_language="cs"),
        EmailRecognizer(supported_language="cs"),
        IbanRecognizer(supported_language="cs"),
        DateRecognizer(supported_language="cs"),
        SpacyRecognizer(supported_language="cs"),
    ]


def _resolve_conflicts(results: List[RecognizerResult]) -> List[RecognizerResult]:
    """
    Keep the highest-scoring result on overlapping spans.

    Mirrors the score-based conflict resolution presidio-anonymizer
    applies before operating on the text (higher score wins; on equal
    scores the longer, earlier span wins).
    """
    ordered = sorted(
        results, key=lambda r: (-r.score, -(r.end - r.start), r.start)
    )
    kept: List[RecognizerResult] = []
    for result in ordered:
        if all(
            result.end <= other.start or result.start >= other.end for other in kept
        ):
            kept.append(result)
    return sorted(kept, key=lambda r: r.start)


def _anonymize(text: str, resolved: List[RecognizerResult]) -> str:
    """Replace resolved entities with numbered <ENTITY_TYPE_N> placeholders."""
    counters: Dict[str, int] = {}
    pieces: List[str] = []
    cursor = 0
    for result in resolved:
        counters[result.entity_type] = counters.get(result.entity_type, 0) + 1
        pieces.append(text[cursor : result.start])
        pieces.append(f"<{result.entity_type}_{counters[result.entity_type]}>")
        cursor = result.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _span_of(snippet: str) -> Tuple[int, int]:
    start = TEXT.index(snippet)
    return start, start + len(snippet)


@pytest.fixture(scope="module")
def czech_analyzer() -> AnalyzerEngine:
    registry = RecognizerRegistry(
        recognizers=_czech_recognizers(), supported_languages=["cs"]
    )
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=NlpEngineMock(nlp_artifacts=_czech_nlp_artifacts(TEXT)),
        supported_languages=["cs"],
    )


@pytest.fixture(scope="module")
def analyzer_results(czech_analyzer) -> List[RecognizerResult]:
    return czech_analyzer.analyze(TEXT, language="cs")


def test_when_no_czech_support_then_czech_entities_are_missed():
    """Before-state: without Czech support only generic entities appear.

    With only the generic recognizers (the pre-change setup), none of
    the Czech entities, no PERSON/LOCATION and no Czech DATE_TIME can be
    detected — and ``850412/0003`` / parts of ``19-123456789/0800`` can
    only ever surface as PHONE_NUMBER false positives.
    """
    registry = RecognizerRegistry(
        recognizers=[
            PhoneRecognizer(supported_language="cs"),
            EmailRecognizer(supported_language="cs"),
            IbanRecognizer(supported_language="cs"),
            DateRecognizer(supported_language="cs"),
        ],
        supported_languages=["cs"],
    )
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=NlpEngineMock(
            nlp_artifacts=NlpArtifacts([], [], [], [], None, "cs")
        ),
        supported_languages=["cs"],
    )
    results = analyzer.analyze(TEXT, language="cs")

    found_entities = {result.entity_type for result in results}
    assert found_entities <= {"PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE", "DATE_TIME"}
    assert "CZ_BIRTH_NUMBER" not in found_entities
    assert "PERSON" not in found_entities

    # The rodné číslo is mis-detected as a phone number (the collision
    # this feature fixes).
    birth_number_span = _span_of("850412/0003")
    assert any(
        result.entity_type == "PHONE_NUMBER"
        and (result.start, result.end) == birth_number_span
        for result in results
    )


@pytest.mark.parametrize("snippet, entity_type", EXPECTED_DETECTIONS)
def test_when_czech_text_analyzed_then_each_field_is_detected(
    analyzer_results, snippet, entity_type
):
    start, end = _span_of(snippet)
    matching = [
        result
        for result in analyzer_results
        if result.entity_type == entity_type
        and result.start == start
        and result.end == end
    ]
    assert matching, (
        f"{entity_type} was not detected on {snippet!r} at ({start}, {end}). "
        f"Got: {[(r.entity_type, TEXT[r.start:r.end]) for r in analyzer_results]}"
    )


def test_when_czech_text_analyzed_then_czech_entities_beat_phone_number(
    analyzer_results,
):
    """The overlap-resolution regression: CZ_* must outscore PHONE_NUMBER."""
    resolved = _resolve_conflicts(analyzer_results)
    phone_numbers = [
        TEXT[result.start : result.end]
        for result in resolved
        if result.entity_type == "PHONE_NUMBER"
    ]
    assert phone_numbers == ["+420 601 234 567"]

    resolved_types = {result.entity_type for result in resolved}
    assert "CZ_BIRTH_NUMBER" in resolved_types
    assert "CZ_BANK_ACCOUNT" in resolved_types


def test_when_czech_text_anonymized_then_output_matches_expected(analyzer_results):
    """The full before -> after scenario from the feature request."""
    resolved = _resolve_conflicts(analyzer_results)
    assert _anonymize(TEXT, resolved) == EXPECTED_ANONYMIZED
