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
