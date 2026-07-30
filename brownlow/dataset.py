from __future__ import annotations

import logging

import pandas as pd

from brownlow.afltables import parse_match_header, parse_match_page
from brownlow.footywire import (
    parse_advanced_stats_page,
    parse_team_roster,
    TEAM_ROSTER_URL_TEMPLATE,
)
from brownlow.http import fetch_url
from brownlow.names import normalize_player_name
from brownlow.teams import canonicalize_team_name, FOOTYWIRE_TEAM_SLUGS

logger = logging.getLogger(__name__)

# One-hot player-position columns. Brownlow voting has a well-documented
# positional bias (midfielders/forwards dominate the medal; defenders/rucks are
# historically underrepresented) that none of the raw-stat features capture.
# A player maps to exactly one bucket (or none, if the position is missing/
# unrecognized -> all four are 0, a reasonable "unknown" encoding for a tree
# model). Sourced per-season from footywire team-roster pages.
POSITION_COLUMNS = [
    "position_forward", "position_midfield", "position_defender", "position_ruck",
]

STAT_COLUMNS = [
    "kicks", "handballs", "disposals", "marks", "goals", "behinds", "hitouts", "tackles",
    "clearances", "contested_possessions", "contested_marks", "goal_assists",
    "score_involvements", "intercepts",
    # 8 additional footywire advanced-stats columns, parsed from the SAME
    # already-fetched ft_match_statistics?advv=Y page as score_involvements/
    # intercepts (no new network requests). All integer counts EXCEPT
    # disposal_efficiency, which is a percentage float (e.g. 73.1). Like SI/ITC
    # they don't exist on footywire's pre-2015 pages and default to 0 there, and
    # they default to 0 when no footywire row joins onto an afltables player.
    "uncontested_possessions", "effective_disposals", "disposal_efficiency",
    "marks_inside_50", "one_percenters", "centre_clearances", "metres_gained",
    "tackles_inside_50",
    # team_margin is appended after the footywire stats so the originals keep
    # their order. It captures the well-documented winning-team bias in Brownlow
    # voting: the player's team's final score minus the opponent's for that match.
    "team_margin",
    # The 4 one-hot position columns are appended last so all earlier columns
    # keep their ordering. They are NOT populated by assemble_match_records --
    # that stays a pure per-match afltables+footywire function -- but by
    # add_position_features as a whole-DataFrame join step.
    *POSITION_COLUMNS,
]

_FOOTYWIRE_ONLY_COLUMNS = (
    "score_involvements", "intercepts",
    "uncontested_possessions", "effective_disposals", "disposal_efficiency",
    "marks_inside_50", "one_percenters", "centre_clearances", "metres_gained",
    "tackles_inside_50",
)
# Columns assemble_match_records populates per row: everything except the
# position one-hots, which are joined on afterward via add_position_features.
_ASSEMBLED_STAT_COLUMNS = [c for c in STAT_COLUMNS if c not in POSITION_COLUMNS]

# Maps a normalized position bucket to its one-hot column.
_POSITION_BUCKET_TO_COLUMN = {
    "forward": "position_forward",
    "midfield": "position_midfield",
    "defender": "position_defender",
    "ruck": "position_ruck",
}
# Raw-string tokens (lowercased) that identify each canonical bucket. Includes
# common abbreviations. For combo positions ("MidfieldForward", "ForwardRuck")
# the EARLIEST-occurring token wins, so the primary (first-listed) role is kept
# -- footywire lists the primary role first -- which keeps the encoding strictly
# one-hot. Prefix aliases (e.g. "mid" vs "midfield") map to the same bucket, so
# an earliest-index tie between them is harmless.
_POSITION_TOKENS = {
    "forward": "forward", "fwd": "forward",
    "midfield": "midfield", "mid": "midfield",
    "defender": "defender", "def": "defender", "back": "defender",
    "ruck": "ruck", "ruc": "ruck",
}


def normalize_position(raw: str | None) -> str | None:
    """Map a raw footywire position string to one of the 4 canonical buckets.

    Returns "forward"/"midfield"/"defender"/"ruck", or ``None`` for a missing,
    empty, or genuinely unrecognized string (the caller then leaves all four
    one-hot columns 0). Case-insensitive; combo positions resolve to the
    primary (earliest-listed) role.
    """
    if not raw:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    best_bucket = None
    best_index = None
    for token, bucket in _POSITION_TOKENS.items():
        i = text.find(token)
        if i != -1 and (best_index is None or i < best_index):
            best_index = i
            best_bucket = bucket
    return best_bucket


def build_position_lookup(
    start_season: int, end_season: int, fetch=fetch_url
) -> dict[tuple[int, str, str], str]:
    """Build ``(season, team_name, normalized_player_name) -> raw position``.

    Fetches every (season, team) roster page for ``start_season..end_season`` x
    all 18 clubs in ``FOOTYWIRE_TEAM_SLUGS`` and parses each with
    ``parse_team_roster``. Called ONCE per backfill/weekly run (not per match),
    so it favours correctness over speed -- but it never re-fetches the same
    (season, team) pair twice. Each fetch/parse is wrapped in try/except: a
    missing or malformed team-season roster is logged and skipped, never
    crashing the whole build (the affected players simply fall back to the
    all-zero "unknown" position encoding downstream).
    """
    lookup: dict[tuple[int, str, str], str] = {}
    fetched: set[tuple[int, str]] = set()
    for season in range(start_season, end_season + 1):
        for team_name, slug in FOOTYWIRE_TEAM_SLUGS.items():
            if (season, team_name) in fetched:
                continue
            fetched.add((season, team_name))
            url = TEAM_ROSTER_URL_TEMPLATE.format(slug=slug, year=season)
            try:
                html = fetch(url)
            except Exception as e:
                logger.warning(
                    "could not fetch roster for %s %s (%s), skipping: %s",
                    team_name, season, url, e,
                )
                continue
            try:
                roster = parse_team_roster(html)
            except Exception as e:
                logger.warning(
                    "could not parse roster for %s %s, skipping: %s", team_name, season, e
                )
                continue
            for entry in roster:
                lookup[(season, team_name, entry["player"])] = entry["position"]
    return lookup


def add_position_features(df: pd.DataFrame, position_lookup: dict) -> pd.DataFrame:
    """Join one-hot position columns onto a fully-assembled DataFrame.

    For each row, looks up ``(season, team, player)`` in ``position_lookup`` and
    sets exactly one of the 4 ``POSITION_COLUMNS`` to 1 (or leaves all 0 when the
    player isn't found or the position is unrecognized). Always adds all 4
    columns -- even on an empty DataFrame or an empty lookup -- so downstream
    ``model.predict(df[STAT_COLUMNS])`` never KeyErrors. Mutates and returns
    ``df``. Logs the match rate, and logs any distinct unrecognized position
    strings once (not per row).
    """
    if len(df) == 0:
        for col in POSITION_COLUMNS:
            df[col] = pd.Series(dtype="int64")
        return df

    buckets = []
    matched = 0
    unrecognized: dict[str, int] = {}
    for season, team, player in zip(df["season"], df["team"], df["player"]):
        raw = position_lookup.get((int(season), team, player))
        bucket = normalize_position(raw)
        if bucket is not None:
            matched += 1
        elif raw:  # found a roster string we couldn't bucket -> track it
            unrecognized[raw] = unrecognized.get(raw, 0) + 1
        buckets.append(bucket)

    for bucket, col in _POSITION_BUCKET_TO_COLUMN.items():
        df[col] = [1 if b == bucket else 0 for b in buckets]

    if unrecognized:
        logger.warning(
            "unrecognized position strings left all-zero (distinct value -> count): %s",
            unrecognized,
        )
    logger.info(
        "position features: %d/%d rows matched a roster position (%d fell back to all-zero)",
        matched, len(df), len(df) - matched,
    )
    return df


def assemble_match_records(
    season: int,
    match_id: str,
    afltables_html: str,
    footywire_html: str | None,
) -> list[dict]:
    header = parse_match_header(afltables_html)
    afltables_rows = parse_match_page(afltables_html)

    footywire_lookup = {}
    if footywire_html:
        for row in parse_advanced_stats_page(footywire_html):
            # footywire's team spelling can be an alias of afltables' (e.g.
            # "Brisbane" vs "Brisbane Lions"); canonicalize it so the join key
            # matches afltables' side, which already uses the canonical spelling.
            key = (canonicalize_team_name(row["team"]), normalize_player_name(row["player"]))
            footywire_lookup[key] = row

    afltables_keys = {(row["team"], normalize_player_name(row["player"])) for row in afltables_rows}
    for key in footywire_lookup:
        if key not in afltables_keys:
            logger.warning("footywire player %s (%s) not found in afltables match %s", key[1], key[0], match_id)

    records = []
    for row in afltables_rows:
        normalized_name = normalize_player_name(row["player"])
        footywire_row = footywire_lookup.get((row["team"], normalized_name))

        # Signed final-score margin from this player's team's perspective:
        # positive if their team won, negative if they lost. Always available
        # from afltables (unlike the footywire-only columns), so it's computed
        # here rather than defaulting to 0 on a footywire miss.
        if row["team"] == header["home_team"]:
            team_margin = header["home_score"] - header["away_score"]
        elif row["team"] == header["away_team"]:
            team_margin = header["away_score"] - header["home_score"]
        else:
            team_margin = 0

        record = {
            "season": season,
            "round": header["round"],
            "date": header["date"],
            "match_id": match_id,
            "team": row["team"],
            "player": normalized_name,
            "brownlow_votes": row["brownlow_votes"],
        }
        for col in _ASSEMBLED_STAT_COLUMNS:
            if col == "team_margin":
                record[col] = team_margin
            elif col in _FOOTYWIRE_ONLY_COLUMNS:
                record[col] = footywire_row[col] if footywire_row else 0
            else:
                record[col] = row[col]
        records.append(record)
    return records


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
