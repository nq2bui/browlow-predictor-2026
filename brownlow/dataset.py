from __future__ import annotations

import logging

import pandas as pd

from brownlow.afltables import parse_match_header, parse_match_page
from brownlow.footywire import parse_advanced_stats_page
from brownlow.names import normalize_player_name

logger = logging.getLogger(__name__)

STAT_COLUMNS = [
    "kicks", "handballs", "disposals", "marks", "goals", "behinds", "hitouts", "tackles",
    "clearances", "contested_possessions", "contested_marks", "goal_assists",
    "score_involvements", "intercepts",
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
            key = (row["team"], normalize_player_name(row["player"]))
            footywire_lookup[key] = row

    afltables_keys = {(row["team"], normalize_player_name(row["player"])) for row in afltables_rows}
    for key in footywire_lookup:
        if key not in afltables_keys:
            logger.warning("footywire player %s (%s) not found in afltables match %s", key[1], key[0], match_id)

    records = []
    for row in afltables_rows:
        normalized_name = normalize_player_name(row["player"])
        footywire_row = footywire_lookup.get((row["team"], normalized_name))

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
            if col in _FOOTYWIRE_ONLY_COLUMNS:
                record[col] = footywire_row[col] if footywire_row else 0
            else:
                record[col] = row[col]
        records.append(record)
    return records


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
