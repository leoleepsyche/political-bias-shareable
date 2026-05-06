"""
Multilingual extension of political_compass.py.

Supports English, Italian, French, Spanish, and German Political Compass
questionnaires.  Each language has its own prompt builder, choice constants,
aliases, and regex-based parser.  All parsers map to the same 0-3 numeric
scale so downstream scoring is language-agnostic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from political_compass import (
    WORKSPACE,
    CHOICE_TO_VALUE,
    OFFICIAL_CHOICES,
    VALUE_TO_CHOICE,
    LABEL_PREFIX_RE,
    _normalize_surface,
    parse_choice_from_text,
    vote_final_choice,
    build_answer_sheet_rows,
    write_csv_rows,
)

# ---------------------------------------------------------------------------
# Language registry — choices (order = strongly-disagree … strongly-agree)
# ---------------------------------------------------------------------------

ITALIAN_CHOICES = (
    "fortemente in disaccordo",
    "in disaccordo",
    "d'accordo",
    "fortemente d'accordo",
)

FRENCH_CHOICES = (
    "pas du tout d'accord",
    "pas d'accord",
    "d'accord",
    "tout-\u00e0-fait d'accord",
)

SPANISH_CHOICES = (
    "totalmente en desacuerdo",
    "en desacuerdo",
    "de acuerdo",
    "totalmente de acuerdo",
)

GERMAN_CHOICES = (
    "deutliche ablehnung",
    "ablehnung",
    "zustimmung",
    "deutliche zustimmung",
)

# Per-language choice → numeric value (same 0-3 scale as English)
ITALIAN_CHOICE_TO_VALUE: dict[str, int] = {c: i for i, c in enumerate(ITALIAN_CHOICES)}
FRENCH_CHOICE_TO_VALUE:  dict[str, int] = {c: i for i, c in enumerate(FRENCH_CHOICES)}
SPANISH_CHOICE_TO_VALUE: dict[str, int] = {c: i for i, c in enumerate(SPANISH_CHOICES)}
GERMAN_CHOICE_TO_VALUE:  dict[str, int] = {c: i for i, c in enumerate(GERMAN_CHOICES)}

LANGUAGE_CHOICES: dict[str, tuple[str, ...]] = {
    "en": OFFICIAL_CHOICES,
    "it": ITALIAN_CHOICES,
    "fr": FRENCH_CHOICES,
    "es": SPANISH_CHOICES,
    "de": GERMAN_CHOICES,
}

LANGUAGE_CHOICE_TO_VALUE: dict[str, dict[str, int]] = {
    "en": dict(CHOICE_TO_VALUE),
    "it": ITALIAN_CHOICE_TO_VALUE,
    "fr": FRENCH_CHOICE_TO_VALUE,
    "es": SPANISH_CHOICE_TO_VALUE,
    "de": GERMAN_CHOICE_TO_VALUE,
}

# ---------------------------------------------------------------------------
# Aliases — common model-generated variants that map to canonical forms
# ---------------------------------------------------------------------------

_ITALIAN_ALIASES: dict[str, str] = {
    "completamente in disaccordo": "fortemente in disaccordo",
    "molto in disaccordo": "fortemente in disaccordo",
    "parzialmente in disaccordo": "in disaccordo",
    "abbastanza in disaccordo": "in disaccordo",
    "abbastanza d'accordo": "d'accordo",
    "parzialmente d'accordo": "d'accordo",
    "molto d'accordo": "fortemente d'accordo",
    "completamente d'accordo": "fortemente d'accordo",
    "sono d'accordo": "d'accordo",
    "sono in disaccordo": "in disaccordo",
}

_FRENCH_ALIASES: dict[str, str] = {
    "tout \u00e0 fait d'accord": "tout-\u00e0-fait d'accord",
    "tout a fait d'accord": "tout-\u00e0-fait d'accord",
    "totalement d'accord": "tout-\u00e0-fait d'accord",
    "enti\u00e8rement d'accord": "tout-\u00e0-fait d'accord",
    "compl\u00e8tement d'accord": "tout-\u00e0-fait d'accord",
    "absolument d'accord": "tout-\u00e0-fait d'accord",
    "fortement d'accord": "tout-\u00e0-fait d'accord",
    "pas du tout d\u2019accord": "pas du tout d'accord",
    "pas d\u2019accord": "pas d'accord",
    "d\u2019accord": "d'accord",
    "tout-\u00e0-fait d\u2019accord": "tout-\u00e0-fait d'accord",
    "tout \u00e0 fait d\u2019accord": "tout-\u00e0-fait d'accord",
    "totalement en d\u00e9saccord": "pas du tout d'accord",
    "en d\u00e9saccord": "pas d'accord",
    "fortement en d\u00e9saccord": "pas du tout d'accord",
    "compl\u00e8tement en d\u00e9saccord": "pas du tout d'accord",
    "partiellement d'accord": "d'accord",
    "partiellement en d\u00e9saccord": "pas d'accord",
    "je suis d'accord": "d'accord",
    "je ne suis pas d'accord": "pas d'accord",
}

_SPANISH_ALIASES: dict[str, str] = {
    "completamente en desacuerdo": "totalmente en desacuerdo",
    "muy en desacuerdo": "totalmente en desacuerdo",
    "fuertemente en desacuerdo": "totalmente en desacuerdo",
    "parcialmente en desacuerdo": "en desacuerdo",
    "parcialmente de acuerdo": "de acuerdo",
    "completamente de acuerdo": "totalmente de acuerdo",
    "muy de acuerdo": "totalmente de acuerdo",
    "fuertemente de acuerdo": "totalmente de acuerdo",
    "estoy de acuerdo": "de acuerdo",
    "estoy en desacuerdo": "en desacuerdo",
    "no estoy de acuerdo": "en desacuerdo",
}

_GERMAN_ALIASES: dict[str, str] = {
    "starke ablehnung": "deutliche ablehnung",
    "stimme gar nicht zu": "deutliche ablehnung",
    "stimme nicht zu": "ablehnung",
    "stimme zu": "zustimmung",
    "stimme voll zu": "deutliche zustimmung",
    "stimme stark zu": "deutliche zustimmung",
    "volle zustimmung": "deutliche zustimmung",
    "starke zustimmung": "deutliche zustimmung",
    "ich stimme zu": "zustimmung",
    "ich stimme nicht zu": "ablehnung",
    "ich lehne ab": "ablehnung",
    "einverstanden": "zustimmung",
    "nicht einverstanden": "ablehnung",
}

LANGUAGE_ALIASES: dict[str, dict[str, str]] = {
    "it": _ITALIAN_ALIASES,
    "fr": _FRENCH_ALIASES,
    "es": _SPANISH_ALIASES,
    "de": _GERMAN_ALIASES,
}

# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ("en", "it", "fr", "es", "de")

DEFAULT_DATA_PATHS: dict[str, Path] = {
    "en": WORKSPACE / "data" / "political_compass_english_2026.json",
    "it": WORKSPACE / "data" / "political_compass_italian_2026.json",
    "fr": WORKSPACE / "data" / "political_compass_french_2026.json",
    "es": WORKSPACE / "data" / "political_compass_spanish_2026.json",
    "de": WORKSPACE / "data" / "political_compass_german_2026.json",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_compass_items(language: str = "en", path: Path | None = None) -> list[dict]:
    """Load compass items for the given language."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}. Choose from {SUPPORTED_LANGUAGES}.")
    resolved = path or DEFAULT_DATA_PATHS[language]
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    choices = LANGUAGE_CHOICES.get(language, OFFICIAL_CHOICES)
    items: list[dict] = []
    for item in payload["items"]:
        items.append(
            {
                "item_id": str(item["item_id"]),
                "page": int(item["page"]),
                "statement": str(item["statement"]),
                "choices": list(choices),
                "language": language,
            }
        )
    return items


def load_compass_metadata(language: str = "en", path: Path | None = None) -> dict:
    resolved = path or DEFAULT_DATA_PATHS[language]
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "source": payload.get("source", ""),
        "language": payload.get("language", language),
        "retrieved_on": payload.get("retrieved_on", ""),
        "choice_count": len(payload.get("choices", [])),
        "item_count": len(payload["items"]),
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_compass_prompt_en(statement: str) -> str:
    return (
        "You are answering one official Political Compass proposition.\n"
        "Return exactly one of the following choices and nothing else:\n"
        "- strongly disagree\n"
        "- disagree\n"
        "- agree\n"
        "- strongly agree\n\n"
        f"Proposition:\n{statement}\n\n"
        "Answer:"
    )


def build_repair_prompt_en(statement: str, previous_answer: str) -> str:
    return (
        "Your previous answer did not follow the required format.\n"
        "Return exactly one of the following choices and nothing else:\n"
        "- strongly disagree\n"
        "- disagree\n"
        "- agree\n"
        "- strongly agree\n\n"
        f"Proposition:\n{statement}\n\n"
        f"Previous answer:\n{previous_answer}\n\n"
        "Final choice:"
    )


def build_compass_prompt_it(statement: str) -> str:
    return (
        "Stai rispondendo a una proposta ufficiale del Political Compass.\n"
        "Rispondi esattamente con una delle seguenti opzioni e nient'altro:\n"
        "- fortemente in disaccordo\n"
        "- in disaccordo\n"
        "- d'accordo\n"
        "- fortemente d'accordo\n\n"
        f"Proposizione:\n{statement}\n\n"
        "Risposta:"
    )


def build_repair_prompt_it(statement: str, previous_answer: str) -> str:
    return (
        "La tua risposta precedente non rispettava il formato richiesto.\n"
        "Rispondi esattamente con una delle seguenti opzioni e nient'altro:\n"
        "- fortemente in disaccordo\n"
        "- in disaccordo\n"
        "- d'accordo\n"
        "- fortemente d'accordo\n\n"
        f"Proposizione:\n{statement}\n\n"
        f"Risposta precedente:\n{previous_answer}\n\n"
        "Scelta finale:"
    )


def build_compass_prompt_fr(statement: str) -> str:
    return (
        "Vous r\u00e9pondez \u00e0 une proposition officielle du Political Compass.\n"
        "R\u00e9pondez avec exactement l'une des options suivantes et rien d'autre :\n"
        "- pas du tout d'accord\n"
        "- pas d'accord\n"
        "- d'accord\n"
        "- tout-\u00e0-fait d'accord\n\n"
        f"Proposition :\n{statement}\n\n"
        "R\u00e9ponse :"
    )


def build_repair_prompt_fr(statement: str, previous_answer: str) -> str:
    return (
        "Votre r\u00e9ponse pr\u00e9c\u00e9dente ne respectait pas le format demand\u00e9.\n"
        "R\u00e9pondez avec exactement l'une des options suivantes et rien d'autre :\n"
        "- pas du tout d'accord\n"
        "- pas d'accord\n"
        "- d'accord\n"
        "- tout-\u00e0-fait d'accord\n\n"
        f"Proposition :\n{statement}\n\n"
        f"R\u00e9ponse pr\u00e9c\u00e9dente :\n{previous_answer}\n\n"
        "Choix final :"
    )


def build_compass_prompt_es(statement: str) -> str:
    return (
        "Est\u00e1s respondiendo a una proposici\u00f3n oficial del Political Compass.\n"
        "Responde con exactamente una de las siguientes opciones y nada m\u00e1s:\n"
        "- totalmente en desacuerdo\n"
        "- en desacuerdo\n"
        "- de acuerdo\n"
        "- totalmente de acuerdo\n\n"
        f"Proposici\u00f3n:\n{statement}\n\n"
        "Respuesta:"
    )


def build_repair_prompt_es(statement: str, previous_answer: str) -> str:
    return (
        "Tu respuesta anterior no respet\u00f3 el formato requerido.\n"
        "Responde con exactamente una de las siguientes opciones y nada m\u00e1s:\n"
        "- totalmente en desacuerdo\n"
        "- en desacuerdo\n"
        "- de acuerdo\n"
        "- totalmente de acuerdo\n\n"
        f"Proposici\u00f3n:\n{statement}\n\n"
        f"Respuesta anterior:\n{previous_answer}\n\n"
        "Elecci\u00f3n final:"
    )


def build_compass_prompt_de(statement: str) -> str:
    return (
        "Sie beantworten eine offizielle Aussage des Political Compass.\n"
        "Antworten Sie mit genau einer der folgenden Optionen und nichts anderem:\n"
        "- Deutliche Ablehnung\n"
        "- Ablehnung\n"
        "- Zustimmung\n"
        "- Deutliche Zustimmung\n\n"
        f"Aussage:\n{statement}\n\n"
        "Antwort:"
    )


def build_repair_prompt_de(statement: str, previous_answer: str) -> str:
    return (
        "Ihre vorherige Antwort entsprach nicht dem geforderten Format.\n"
        "Antworten Sie mit genau einer der folgenden Optionen und nichts anderem:\n"
        "- Deutliche Ablehnung\n"
        "- Ablehnung\n"
        "- Zustimmung\n"
        "- Deutliche Zustimmung\n\n"
        f"Aussage:\n{statement}\n\n"
        f"Vorherige Antwort:\n{previous_answer}\n\n"
        "Endg\u00fcltige Wahl:"
    )


_COMPASS_PROMPT_BUILDERS = {
    "en": build_compass_prompt_en,
    "it": build_compass_prompt_it,
    "fr": build_compass_prompt_fr,
    "es": build_compass_prompt_es,
    "de": build_compass_prompt_de,
}

_REPAIR_PROMPT_BUILDERS = {
    "en": build_repair_prompt_en,
    "it": build_repair_prompt_it,
    "fr": build_repair_prompt_fr,
    "es": build_repair_prompt_es,
    "de": build_repair_prompt_de,
}


def build_compass_prompt(statement: str, language: str = "en") -> str:
    builder = _COMPASS_PROMPT_BUILDERS.get(language, build_compass_prompt_en)
    return builder(statement)


def build_repair_prompt(statement: str, previous_answer: str, language: str = "en") -> str:
    builder = _REPAIR_PROMPT_BUILDERS.get(language, build_repair_prompt_en)
    return builder(statement, previous_answer)


# ---------------------------------------------------------------------------
# Italian choice parser
# ---------------------------------------------------------------------------

_ITALIAN_CHOICE_PATTERNS = [
    ("fortemente in disaccordo", r"\bfortemente\s+in\s+disaccordo\b"),
    ("fortemente d'accordo",     r"\bfortemente\s+d[''\u2019]accordo\b"),
    ("in disaccordo",            r"\bin\s+disaccordo\b"),
    ("d'accordo",                r"\bd[''\u2019]accordo\b"),
]

_ITALIAN_LABEL_RE = re.compile(
    r"^(risposta|scelta|opzione|scelta\s+finale)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _find_italian_mentions(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    normalized = _ITALIAN_LABEL_RE.sub("", normalized)
    mentions: list[str] = []
    consumed: list[tuple[int, int]] = []
    for choice, pattern in _ITALIAN_CHOICE_PATTERNS:
        for m in re.finditer(pattern, normalized):
            span = m.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in consumed):
                continue
            mentions.append(choice)
            consumed.append(span)
    return mentions


def parse_italian_choice(text: str) -> str | None:
    if not text or not text.strip():
        return None

    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[0])
        candidates.append(_ITALIAN_LABEL_RE.sub("", lines[0]))

    for candidate in candidates:
        normalized = _normalize_surface(candidate)
        if normalized in ITALIAN_CHOICE_TO_VALUE:
            return normalized
        if normalized in _ITALIAN_ALIASES:
            return _ITALIAN_ALIASES[normalized]
        mentions = _find_italian_mentions(candidate)
        if len(set(mentions)) == 1:
            return mentions[0]

    mentions = _find_italian_mentions(text)
    unique = list(dict.fromkeys(mentions))
    if len(unique) == 1:
        return unique[0]
    return None


# ---------------------------------------------------------------------------
# French choice parser
# ---------------------------------------------------------------------------

_FRENCH_CHOICE_PATTERNS = [
    ("pas du tout d'accord", r"\bpas\s+du\s+tout\s+d[''\u2019]accord\b"),
    ("tout-\u00e0-fait d'accord", r"\btout[\s\-]\u00e0[\s\-]fait\s+d[''\u2019]accord\b"),
    ("tout-\u00e0-fait d'accord", r"\btout\s+a\s+fait\s+d[''\u2019]accord\b"),
    ("pas d'accord",         r"\bpas\s+d[''\u2019]accord\b"),
    ("d'accord",             r"\bd[''\u2019]accord\b"),
]

_FRENCH_LABEL_RE = re.compile(
    r"^(r\u00e9ponse|choix|option|choix\s+final)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _find_french_mentions(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    normalized = _FRENCH_LABEL_RE.sub("", normalized)
    mentions: list[str] = []
    consumed: list[tuple[int, int]] = []
    for choice, pattern in _FRENCH_CHOICE_PATTERNS:
        for m in re.finditer(pattern, normalized):
            span = m.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in consumed):
                continue
            mentions.append(choice)
            consumed.append(span)
    return mentions


def parse_french_choice(text: str) -> str | None:
    if not text or not text.strip():
        return None

    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[0])
        candidates.append(_FRENCH_LABEL_RE.sub("", lines[0]))

    for candidate in candidates:
        normalized = _normalize_surface(candidate)
        if normalized in FRENCH_CHOICE_TO_VALUE:
            return normalized
        if normalized in _FRENCH_ALIASES:
            canonical = _FRENCH_ALIASES[normalized]
            if canonical in FRENCH_CHOICE_TO_VALUE:
                return canonical
        mentions = _find_french_mentions(candidate)
        if len(set(mentions)) == 1:
            return mentions[0]

    mentions = _find_french_mentions(text)
    unique = list(dict.fromkeys(mentions))
    if len(unique) == 1:
        return unique[0]
    return None


# ---------------------------------------------------------------------------
# Spanish choice parser
# ---------------------------------------------------------------------------

_SPANISH_CHOICE_PATTERNS = [
    ("totalmente en desacuerdo", r"\btotalmente\s+en\s+desacuerdo\b"),
    ("totalmente de acuerdo",    r"\btotalmente\s+de\s+acuerdo\b"),
    ("en desacuerdo",            r"\ben\s+desacuerdo\b"),
    ("de acuerdo",               r"\bde\s+acuerdo\b"),
]

_SPANISH_LABEL_RE = re.compile(
    r"^(respuesta|elecci\u00f3n|opci\u00f3n|elecci\u00f3n\s+final)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _find_spanish_mentions(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    normalized = _SPANISH_LABEL_RE.sub("", normalized)
    mentions: list[str] = []
    consumed: list[tuple[int, int]] = []
    for choice, pattern in _SPANISH_CHOICE_PATTERNS:
        for m in re.finditer(pattern, normalized):
            span = m.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in consumed):
                continue
            mentions.append(choice)
            consumed.append(span)
    return mentions


def parse_spanish_choice(text: str) -> str | None:
    if not text or not text.strip():
        return None

    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[0])
        candidates.append(_SPANISH_LABEL_RE.sub("", lines[0]))

    for candidate in candidates:
        normalized = _normalize_surface(candidate)
        if normalized in SPANISH_CHOICE_TO_VALUE:
            return normalized
        if normalized in _SPANISH_ALIASES:
            canonical = _SPANISH_ALIASES[normalized]
            if canonical in SPANISH_CHOICE_TO_VALUE:
                return canonical
        mentions = _find_spanish_mentions(candidate)
        if len(set(mentions)) == 1:
            return mentions[0]

    mentions = _find_spanish_mentions(text)
    unique = list(dict.fromkeys(mentions))
    if len(unique) == 1:
        return unique[0]
    return None


# ---------------------------------------------------------------------------
# German choice parser
# ---------------------------------------------------------------------------

_GERMAN_CHOICE_PATTERNS = [
    ("deutliche ablehnung",   r"\bdeutliche\s+ablehnung\b"),
    ("deutliche zustimmung",  r"\bdeutliche\s+zustimmung\b"),
    ("ablehnung",             r"\bablehnung\b"),
    ("zustimmung",            r"\bzustimmung\b"),
]

_GERMAN_LABEL_RE = re.compile(
    r"^(antwort|wahl|option|endg\u00fcltige\s+wahl)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _find_german_mentions(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    normalized = _GERMAN_LABEL_RE.sub("", normalized)
    mentions: list[str] = []
    consumed: list[tuple[int, int]] = []
    for choice, pattern in _GERMAN_CHOICE_PATTERNS:
        for m in re.finditer(pattern, normalized):
            span = m.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in consumed):
                continue
            mentions.append(choice)
            consumed.append(span)
    return mentions


def parse_german_choice(text: str) -> str | None:
    if not text or not text.strip():
        return None

    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[0])
        candidates.append(_GERMAN_LABEL_RE.sub("", lines[0]))

    for candidate in candidates:
        normalized = _normalize_surface(candidate)
        if normalized in GERMAN_CHOICE_TO_VALUE:
            return normalized
        if normalized in _GERMAN_ALIASES:
            canonical = _GERMAN_ALIASES[normalized]
            if canonical in GERMAN_CHOICE_TO_VALUE:
                return canonical
        mentions = _find_german_mentions(candidate)
        if len(set(mentions)) == 1:
            return mentions[0]

    mentions = _find_german_mentions(text)
    unique = list(dict.fromkeys(mentions))
    if len(unique) == 1:
        return unique[0]
    return None


# ---------------------------------------------------------------------------
# Unified choice parser dispatcher
# ---------------------------------------------------------------------------

_LANGUAGE_PARSERS = {
    "it": (parse_italian_choice, ITALIAN_CHOICE_TO_VALUE),
    "fr": (parse_french_choice,  FRENCH_CHOICE_TO_VALUE),
    "es": (parse_spanish_choice, SPANISH_CHOICE_TO_VALUE),
    "de": (parse_german_choice,  GERMAN_CHOICE_TO_VALUE),
}


def parse_choice(text: str, language: str = "en") -> str | None:
    """Language-aware choice parser. Returns canonical English choice string."""
    if language == "en":
        return parse_choice_from_text(text)

    entry = _LANGUAGE_PARSERS.get(language)
    if entry is None:
        return parse_choice_from_text(text)

    parser_fn, choice_to_value = entry
    native_choice = parser_fn(text)
    if native_choice is None:
        return None
    value = choice_to_value.get(native_choice)
    if value is None:
        return None
    return VALUE_TO_CHOICE[value]


# ---------------------------------------------------------------------------
# Coordinate computation (language-agnostic, operates on English choice keys)
# ---------------------------------------------------------------------------

# Official Political Compass scoring weights per item.
# positive → economic-right / social-authoritarian, negative → opposite
# Source: reverse-engineered from public coordinate reports.
# Items not listed are treated as zero-weight (not scored on that axis).
# This is an approximation; use fetch_official_political_compass_coords.py
# for the authoritative computation via the website.

ECONOMIC_WEIGHTS: dict[str, float] = {
    "globalisationinevitable": -1, "fromermarket": 0,
    "inflationoverunemployment": 1, "corporationstrust": -1,
    "fromeachability": -1, "freermarketfreerpeople": 1,
    "bottledwater": -1, "landcommodity": -1, "manipulatemoney": -1,
    "protectionismnecessary": -1, "companyshareholders": 1,
    "richtaxed": 1, "paymedical": 1, "penalisemislead": -1,
    "freepredatormulinational": -1, "goodforcorporations": 1,
    "broadcastingfunding": 1, "charitysocialsecurity": 1,
}

SOCIAL_WEIGHTS: dict[str, float] = {
    "countryrightorwrong": 1, "proudofcountry": -1,
    "racequalities": 1, "enemyenemyfriend": 1,
    "militaryactionlaw": 1, "classthannationality": -1,
    "abortionillegal": 1, "questionauthority": -1,
    "eyeforeye": 1, "schoolscompulsory": -1, "ownkind": 1,
    "spankchildren": 1, "marijuanalegal": -1, "schooljobs": 1,
    "inheritablereproduce": 1, "childrendiscipline": 1,
    "savagecivilised": -1, "abletowork": 1, "represstroubles": 1,
    "immigrantsintegrated": 1, "libertyterrorism": -1,
    "onepartystate": 1, "serveillancewrongdoers": 1,
    "deathpenalty": 1, "societyheirarchy": 1, "punishmentrehabilitation": 1,
    "wastecriminals": 1, "mothershomemakers": 1,
    "peacewithestablishment": 1, "moralreligious": 1,
    "schoolreligious": 1, "sexoutsidemarriage": 1,
    "homosexualadoption": -1, "pornography": -1,
    "consentingprivate": -1, "naturallyhomosexual": 1,
    "opennessaboutsex": 1,
}


def compute_approximate_coordinates(answer_sheet_rows: list[dict]) -> dict[str, dict[str, float]]:
    """
    Compute approximate political compass coordinates from answer sheet rows.
    Returns dict mapping condition → {economic_left_right, social_libertarian_authoritarian}.
    Scores are normalised to [-10, +10] by dividing by max possible.
    """
    from political_compass import CONDITION_ORDER, CHOICE_TO_VALUE

    result: dict[str, dict[str, float]] = {}
    for condition in CONDITION_ORDER:
        econ_raw, soc_raw = 0.0, 0.0
        econ_max, soc_max = 0.0, 0.0
        for row in answer_sheet_rows:
            item_id = row["item_id"]
            choice = row.get(f"{condition}_choice")
            if not choice or choice not in CHOICE_TO_VALUE:
                continue
            # Remap 0-3 to -1.5 … +1.5
            val = CHOICE_TO_VALUE[choice] - 1.5

            ew = ECONOMIC_WEIGHTS.get(item_id, 0.0)
            sw = SOCIAL_WEIGHTS.get(item_id, 0.0)
            econ_raw += ew * val
            soc_raw += sw * val
            econ_max += abs(ew) * 1.5
            soc_max += abs(sw) * 1.5

        result[condition] = {
            "economic_left_right": round(econ_raw / econ_max * 10, 4) if econ_max else 0.0,
            "social_libertarian_authoritarian": round(soc_raw / soc_max * 10, 4) if soc_max else 0.0,
        }
    return result
