from __future__ import annotations

import logging

import pandas as pd

from brownlow.afltables import parse_match_header, parse_match_page
from brownlow.footywire import parse_advanced_stats_page
from brownlow.names import normalize_player_name
from brownlow.teams import canonicalize_team_name

logger = logging.getLogger(__name__)

STAT_COLUMNS = [
    "kicks", "handballs", "disposals", "marks", "goals", "behinds", "hitouts", "tackles",
    "clearances", "contested_possessions", "contested_marks", "goal_assists",
    "score_involvements", "intercepts",
    # team_margin is appended last so the existing 14 columns keep their order.
    # It captures the well-documented winning-team bias in Brownlow voting:
    # the player's team's final score minus the opponent's for that match.
    "team_margin",
]

_FOOTYWIRE_ONLY_COLUMNS = ("score_involvements", "intercepts")


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
        for col in STAT_COLUMNS:
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
