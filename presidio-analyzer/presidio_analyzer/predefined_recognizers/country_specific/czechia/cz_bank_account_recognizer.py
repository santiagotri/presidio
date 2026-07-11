from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class CzBankAccountRecognizer(PatternRecognizer):
    """
    Recognizes Czech domestic bank account numbers.

    Czech account numbers use the national format
    ``[prefix-]number/bank_code``: an optional prefix of up to 6 digits,
    an account number of 2-10 digits and a mandatory 4-digit bank code
    (e.g. 0800 Česká spořitelna, 0100 Komerční banka).

    Legal basis: vyhláška ČNB č. 169/2011 Sb., o pravidlech tvorby čísla
    účtu v platebním styku. Data protection: GDPR Art. 4 Nr. 1,
    zákon č. 110/2019 Sb.

    Format:
        - Prefix (optional): 2-6 digits followed by a hyphen
        - Account number: 2-10 digits
        - Slash and a 4-digit bank code

    Checksum: both the prefix and the account number carry a weighted
    mod 11 check (weights 1, 2, 4, 8, 5, 10, 9, 7, 3, 6 applied from the
    rightmost digit). A full checksum pass promotes the match to
    MAX_SCORE; a failure keeps the base pattern score so that context
    words ("účet", "účtu", ...) drive the final confidence.

    Examples (fictitious): 19-2000145399/0800, 2000145399/0800

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    COUNTRY_CODE = "cz"

    # Weights from vyhláška č. 169/2011 Sb., applied right-to-left.
    _WEIGHTS = (1, 2, 4, 8, 5, 10, 9, 7, 3, 6)

    PATTERNS = [
        Pattern(
            "Czech bank account (prefix-number/bank code)",
            r"(?<![\w/-])(?:\d{2,6}-)?\d{2,10}/\d{4}(?![\w/-])",
            0.4,
        ),
    ]

    CONTEXT = [
        "účet",
        "účtu",
        "účtem",
        "účtě",
        "bankovní",
        "banka",
        "banky",
        "bance",
        "sporožiro",
        "platba",
        "platby",
        "ucet",
        "uctu",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "cs",
        supported_entity: str = "CZ_BANK_ACCOUNT",
        name: Optional[str] = None,
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name,
        )

    def _mod11(self, digits: str) -> bool:
        total = sum(
            int(digit) * weight
            for digit, weight in zip(reversed(digits), self._WEIGHTS)
        )
        return total % 11 == 0

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        """
        Validate the account number using the ČNB weighted mod 11 check.

        Both the optional prefix and the account number must pass the
        weighted checksum for the match to be promoted to MAX_SCORE. A
        checksum failure returns ``None`` (keep the pattern score and let
        context words decide) because typos and formatted fragments are
        common in free text; only structurally impossible values
        (all-zero account, bank code 0000) return ``False``.

        :param pattern_text: the text to validate (prefix-number/bank code)
        :return: True if all checksums pass, False if structurally
                 invalid, None to fall back to the pattern score
        """
        pattern_text = pattern_text.strip()

        account_part, _, bank_code = pattern_text.partition("/")
        if len(bank_code) != 4 or not bank_code.isdigit():
            return False
        if bank_code == "0000":
            return False

        prefix, _, number = account_part.rpartition("-")
        if not number.isdigit() or (prefix and not prefix.isdigit()):
            return False

        # An account number of all zeros is never assigned.
        if number.lstrip("0") == "":
            return False

        if self._mod11(number) and (not prefix or self._mod11(prefix)):
            return True

        return None

    def invalidate_result(self, pattern_text: str) -> Optional[bool]:
        """
        Invalidate matches that are far more likely to be dates.

        A bare ``d/yyyy`` or ``dd/yyyy`` with a plausible day/month value
        and a 19xx/20xx "bank code" (e.g. 04/2023) is treated as a date
        fragment, not an account number.

        :param pattern_text: the text to check
        :return: True if the match should be invalidated, None otherwise
        """
        account_part, _, bank_code = pattern_text.partition("/")
        if (
            "-" not in account_part
            and len(account_part) <= 2
            and account_part.isdigit()
            and int(account_part) <= 31
            and bank_code[:2] in ("19", "20")
        ):
            return True
        return None
