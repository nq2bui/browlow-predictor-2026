"""Team metadata: 3-letter codes and brand colors for AFL clubs.

Team-name strings are the exact values afltables.com uses (confirmed from a
real backfill run against 2025 season data). Brand colors are sourced from
the sibling afl-tipster project's ``TEAMS`` object.

The dashboard uses this to render each club's logo and a brand-color accent.
Team names are NOT canonicalized across data sources (see CLAUDE.md "Known
Limitations"), so ``get_team_info`` degrades gracefully on an unrecognized
name rather than raising.
"""

TEAM_INFO = {
    "Adelaide":               {"code": "ADE", "color": "#002B5C"},
    "Brisbane Lions":         {"code": "BRI", "color": "#A30046"},
    "Carlton":                {"code": "CAR", "color": "#0E1E2D"},
    "Collingwood":            {"code": "COL", "color": "#000000"},
    "Essendon":               {"code": "ESS", "color": "#CC2031"},
    "Fremantle":              {"code": "FRE", "color": "#2A1A5E"},
    "Geelong":                {"code": "GEE", "color": "#1C3C63"},
    "Gold Coast":             {"code": "GCS", "color": "#CC2031"},
    "Greater Western Sydney": {"code": "GWS", "color": "#FF7900"},
    "Hawthorn":               {"code": "HAW", "color": "#4D2004"},
    "Melbourne":              {"code": "MEL", "color": "#CC2031"},
    "North Melbourne":        {"code": "NTH", "color": "#013B9F"},
    "Port Adelaide":          {"code": "POR", "color": "#008AAB"},
    "Richmond":               {"code": "RIC", "color": "#FFD200"},
    "St Kilda":               {"code": "STK", "color": "#ED0F05"},
    "Sydney":                 {"code": "SYD", "color": "#ED171F"},
    "Western Bulldogs":       {"code": "WBD", "color": "#014896"},
    "West Coast":             {"code": "WCE", "color": "#003087"},
}

_FALLBACK = {"code": "", "color": "#666666"}

# footywire spells two clubs differently from afltables. Verified by diffing the
# full real team-name sets from both sites across the whole 2025 season: these
# are the ONLY two aliases -- the other 16 clubs are spelled identically by both.
# Keys are footywire's spelling; values are the canonical afltables spelling
# (the one TEAM_INFO / the join keys use). Add future aliases here as needed.
_TEAM_NAME_ALIASES = {
    "Brisbane": "Brisbane Lions",
    "GWS": "Greater Western Sydney",
}


def canonicalize_team_name(name: str) -> str:
    """Map a known team-name alias to its canonical (afltables) spelling.

    footywire and afltables disagree on exactly two club names; this normalizes
    footywire's spelling onto afltables' so both sides of every afltables↔
    footywire join compare equal. Any name not in the alias map (the 16
    identically-spelled clubs, or anything unrecognized) is returned unchanged,
    so this is always a safe no-op passthrough rather than a lookup that fails.
    """
    return _TEAM_NAME_ALIASES.get(name, name)


def get_team_info(team_name: str) -> dict:
    """Return ``{"code", "color"}`` for ``team_name``.

    Falls back to an empty code and a neutral grey for any name not in
    ``TEAM_INFO`` so the dashboard never crashes on an unexpected team.
    """
    return TEAM_INFO.get(team_name, _FALLBACK)
