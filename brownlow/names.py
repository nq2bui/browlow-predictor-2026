def normalize_player_name(raw: str) -> str:
    raw = raw.strip()
    if "," in raw:
        surname, first = (part.strip() for part in raw.split(",", 1))
        first_initial = first[0] if first else ""
    else:
        parts = raw.split()
        first_initial = parts[0][0] if parts else ""
        surname = " ".join(parts[1:]) if len(parts) > 1 else raw
    return f"{first_initial}. {surname}".strip()


def join_key(normalized_name: str) -> str:
    """Loosened comparison key for matching a ``normalize_player_name`` output
    across afltables and footywire.

    The two sites spell some surnames differently even after normalization:
    afltables strips apostrophes and lowercases some prefixes in its own name
    text (e.g. "OSullivan, Finn" -> "F. OSullivan", "de Goey" stays lowercase),
    while footywire keeps the apostrophe and its own capitalization (e.g.
    "Finn O'Sullivan" -> "F. O'Sullivan", "Jordan De Goey" -> "J. De Goey").
    Folding out apostrophes, periods, spaces and case reconciles these for
    matching WITHOUT changing the displayed name (only this key, never
    ``normalize_player_name``'s output, is used for the join).
    """
    return normalized_name.replace("'", "").replace(".", "").replace(" ", "").lower()
