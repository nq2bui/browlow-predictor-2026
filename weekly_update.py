# weekly_update.py
import logging

import pandas as pd

from brownlow.afltables import (
    SEASON_INDEX_URL_TEMPLATE,
    SEASON_RESULTS_URL_TEMPLATE,
    list_season_match_urls,
    match_id_from_url,
    parse_match_header,
)
from brownlow.footywire import SEASON_MATCH_LIST_URL_TEMPLATE, MATCH_STATS_URL_TEMPLATE, list_season_match_ids
from brownlow.dataset import assemble_match_records, add_position_features, build_position_lookup
from brownlow.model import load_model
from brownlow.weekly import accumulate_season_votes, per_round_votes
from brownlow.dashboard import render_leaderboard, render_round_matrix
from brownlow.odds import fetch_odds_page_html, parse_brownlow_odds
from brownlow.http import fetch_url
from brownlow.teams import canonicalize_team_name

logger = logging.getLogger(__name__)

CURRENT_SEASON = 2026
MODEL_PATH = "model.txt"
OUTPUT_PATH = "index.html"
MATRIX_OUTPUT_PATH = "rounds.html"


def build_current_season_dataframe(season: int, fetch=fetch_url) -> pd.DataFrame:
    try:
        index_html = fetch(SEASON_INDEX_URL_TEMPLATE.format(year=season))
    except Exception as e:
        # The Brownlow round-by-round page only appears once a season concludes,
        # so for an in-progress season it 404s. Fall back to the general
        # season-results page, which exists mid-season and carries the same
        # match-link format that list_season_match_urls already parses.
        logger.warning(
            "could not fetch Brownlow round-by-round index for %s (%s), "
            "falling back to season-results page",
            season,
            e,
        )
        try:
            index_html = fetch(SEASON_RESULTS_URL_TEMPLATE.format(year=season))
        except Exception as e2:
            logger.error(
                "could not fetch season-results page for %s either, aborting "
                "weekly update: %s",
                season,
                e2,
            )
            return pd.DataFrame()

    match_urls = list_season_match_urls(index_html)

    try:
        footywire_matches = list_season_match_ids(fetch(SEASON_MATCH_LIST_URL_TEMPLATE.format(year=season)))
    except Exception as e:
        logger.warning("could not fetch footywire match list for %s, continuing without it: %s", season, e)
        footywire_matches = []
    footywire_html_by_teams = {}
    for match in footywire_matches:
        # Key on the canonical (afltables) team spelling so the lookup below --
        # done with afltables' header team names -- matches even for the clubs
        # footywire spells differently (Brisbane, GWS).
        key = (
            canonicalize_team_name(match["home_team"]),
            canonicalize_team_name(match["away_team"]),
        )
        try:
            footywire_html_by_teams[key] = fetch(
                MATCH_STATS_URL_TEMPLATE.format(mid=match["mid"])
            )
        except Exception as e:
            logger.warning("could not fetch footywire stats for mid=%s, skipping: %s", match["mid"], e)

    all_records = []
    for match_url in match_urls:
        match_id = match_id_from_url(match_url)
        try:
            afltables_html = fetch(match_url)
        except Exception as e:
            logger.warning("could not fetch afltables match %s, skipping: %s", match_url, e)
            continue

        try:
            header = parse_match_header(afltables_html)
            footywire_html = footywire_html_by_teams.get((header["home_team"], header["away_team"]))
            if footywire_html is None:
                logger.warning(
                    "no footywire match found for afltables team pairing (%s, %s) "
                    "in match %s -- footywire-derived features will be 0 for this "
                    "match (possible team-name mismatch between the two sources)",
                    header["home_team"],
                    header["away_team"],
                    match_url,
                )
            all_records.extend(assemble_match_records(season, match_id, afltables_html, footywire_html))
        except Exception as e:
            logger.warning("could not parse/assemble afltables match %s, skipping: %s", match_url, e)
            continue

    return pd.DataFrame(all_records)


def main():
    logging.basicConfig(level=logging.INFO)

    season_df = build_current_season_dataframe(CURRENT_SEASON)
    if season_df.empty:
        logger.error("no 2026 match data available yet, leaving existing index.html untouched")
        return

    # Join current-season player positions (18 roster fetches -- only this
    # season's rosters are needed, not the full history). Built once here and
    # joined onto season_df before scoring. If the build fails entirely (e.g.
    # footywire down), degrade to all-zero position columns for every row so the
    # weekly update still runs -- the model functioned without this feature
    # before it existed.
    try:
        position_lookup = build_position_lookup(CURRENT_SEASON, CURRENT_SEASON)
    except Exception as e:
        logger.warning(
            "could not build position lookup, scoring with all-zero position "
            "features for every row: %s",
            e,
        )
        position_lookup = {}
    season_df = add_position_features(season_df, position_lookup)

    model = load_model(MODEL_PATH)
    leaderboard = accumulate_season_votes(model, season_df)
    # render_leaderboard shows only the top 20; compute per-round detail for
    # exactly those players (same head(20) convention) to keep it cheap.
    top_20_players = leaderboard.head(20)["player"].tolist()
    round_votes = per_round_votes(model, season_df, top_20_players)

    # Sportsbet Brownlow odds are a nice-to-have display feature, not core to the
    # prediction. A fetch failure (SPA render timeout, page change, missing
    # browser) must never crash the weekly update -- degrade gracefully to no
    # odds, same pattern as the per-match try/except elsewhere in the pipeline.
    try:
        odds = parse_brownlow_odds(fetch_odds_page_html())
    except Exception as e:
        logger.warning("could not fetch/parse Sportsbet Brownlow odds, continuing without them: %s", e)
        odds = []

    render_leaderboard(leaderboard, round_votes, odds, OUTPUT_PATH, CURRENT_SEASON)
    logger.info("wrote updated leaderboard to %s (%d players)", OUTPUT_PATH, len(leaderboard))

    # Second page: the same round_votes rendered as an all-20 round-by-round
    # matrix (reuses the round_votes DataFrame already computed above).
    render_round_matrix(leaderboard, round_votes, MATRIX_OUTPUT_PATH, CURRENT_SEASON)
    logger.info("wrote round-by-round matrix to %s", MATRIX_OUTPUT_PATH)


if __name__ == "__main__":
    main()
