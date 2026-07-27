# weekly_update.py
import logging

import pandas as pd

from brownlow.afltables import SEASON_INDEX_URL_TEMPLATE, list_season_match_urls, parse_match_header
from brownlow.footywire import SEASON_MATCH_LIST_URL_TEMPLATE, MATCH_STATS_URL_TEMPLATE, list_season_match_ids
from brownlow.dataset import assemble_match_records
from brownlow.model import load_model
from brownlow.weekly import accumulate_season_votes
from brownlow.dashboard import render_leaderboard
from brownlow.http import fetch_url

logger = logging.getLogger(__name__)

CURRENT_SEASON = 2026
MODEL_PATH = "model.txt"
OUTPUT_PATH = "index.html"


def build_current_season_dataframe(season: int, fetch=fetch_url) -> pd.DataFrame:
    try:
        index_html = fetch(SEASON_INDEX_URL_TEMPLATE.format(year=season))
    except Exception:
        logger.error("could not fetch season index for %s, aborting weekly update", season)
        return pd.DataFrame()

    match_urls = list_season_match_urls(index_html)

    try:
        footywire_matches = list_season_match_ids(fetch(SEASON_MATCH_LIST_URL_TEMPLATE.format(year=season)))
    except Exception:
        logger.warning("could not fetch footywire match list for %s, continuing without it", season)
        footywire_matches = []
    footywire_html_by_teams = {}
    for match in footywire_matches:
        try:
            footywire_html_by_teams[(match["home_team"], match["away_team"])] = fetch(
                MATCH_STATS_URL_TEMPLATE.format(mid=match["mid"])
            )
        except Exception:
            logger.warning("could not fetch footywire stats for mid=%s", match["mid"])

    all_records = []
    for match_url in match_urls:
        match_id = match_url.rsplit("/", 1)[1].replace(".html", "")
        try:
            afltables_html = fetch(match_url)
        except Exception:
            logger.warning("could not fetch afltables match %s, skipping", match_url)
            continue
        header = parse_match_header(afltables_html)
        footywire_html = footywire_html_by_teams.get((header["home_team"], header["away_team"]))
        all_records.extend(assemble_match_records(season, match_id, afltables_html, footywire_html))

    return pd.DataFrame(all_records)


def main():
    logging.basicConfig(level=logging.INFO)

    season_df = build_current_season_dataframe(CURRENT_SEASON)
    if season_df.empty:
        logger.error("no 2026 match data available yet, leaving existing index.html untouched")
        return

    model = load_model(MODEL_PATH)
    leaderboard = accumulate_season_votes(model, season_df)
    render_leaderboard(leaderboard, OUTPUT_PATH)
    logger.info("wrote updated leaderboard to %s (%d players)", OUTPUT_PATH, len(leaderboard))


if __name__ == "__main__":
    main()
