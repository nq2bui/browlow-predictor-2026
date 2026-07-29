import argparse
import logging
from pathlib import Path

import pandas as pd

from brownlow.afltables import (
    SEASON_INDEX_URL_TEMPLATE,
    SEASON_RESULTS_URL_TEMPLATE,
    list_season_match_urls,
    match_id_from_url,
    parse_match_header,
)
from brownlow.footywire import (
    SEASON_MATCH_LIST_URL_TEMPLATE,
    MATCH_STATS_URL_TEMPLATE,
    list_season_match_ids,
)
from brownlow.dataset import assemble_match_records
from brownlow.http import fetch_url
from brownlow.teams import canonicalize_team_name

logger = logging.getLogger(__name__)


def backfill_seasons(start_season: int, end_season: int, fetch=fetch_url) -> pd.DataFrame:
    all_records = []
    for season in range(start_season, end_season + 1):
        try:
            index_html = fetch(SEASON_INDEX_URL_TEMPLATE.format(year=season))
        except Exception as e:
            # The Brownlow round-by-round page only appears once a season
            # concludes; fall back to the general season-results page (same
            # match-link format) so a very recent or in-progress season can
            # still be backfilled. Mirrors weekly_update.py's fallback.
            logger.warning(
                "could not fetch Brownlow round-by-round index for %s (%s), "
                "falling back to season-results page",
                season,
                e,
            )
            try:
                index_html = fetch(SEASON_RESULTS_URL_TEMPLATE.format(year=season))
            except Exception as e2:
                logger.warning(
                    "could not fetch season-results page for %s either, "
                    "skipping season: %s",
                    season,
                    e2,
                )
                continue

        match_urls = list_season_match_urls(index_html)

        try:
            footywire_matches = list_season_match_ids(
                fetch(SEASON_MATCH_LIST_URL_TEMPLATE.format(year=season))
            )
        except Exception as e:
            logger.warning(
                "could not fetch footywire match list for %s, continuing without it: %s", season, e
            )
            footywire_matches = []

        footywire_html_by_teams = {}
        for match in footywire_matches:
            # Key on the canonical (afltables) team spelling so the lookup below
            # -- done with afltables' header team names -- matches even for the
            # clubs footywire spells differently (Brisbane, GWS).
            key = (
                canonicalize_team_name(match["home_team"]),
                canonicalize_team_name(match["away_team"]),
            )
            try:
                footywire_html_by_teams[key] = fetch(
                    MATCH_STATS_URL_TEMPLATE.format(mid=match["mid"])
                )
            except Exception as e:
                logger.warning("could not fetch footywire stats for mid=%s: %s", match["mid"], e)

        for match_url in match_urls:
            match_id = match_id_from_url(match_url)
            try:
                afltables_html = fetch(match_url)
            except Exception as e:
                logger.warning("could not fetch afltables match %s, skipping: %s", match_url, e)
                continue

            try:
                header = parse_match_header(afltables_html)
                footywire_html = footywire_html_by_teams.get(
                    (header["home_team"], header["away_team"])
                )
                if footywire_html is None:
                    logger.warning(
                        "no footywire match found for afltables team pairing (%s, %s) "
                        "in match %s -- footywire-derived features will be 0 for this "
                        "match (possible team-name mismatch between the two sources)",
                        header["home_team"],
                        header["away_team"],
                        match_url,
                    )
                match_records = assemble_match_records(
                    season, match_id, afltables_html, footywire_html
                )
            except Exception as e:
                logger.warning(
                    "could not parse/assemble afltables match %s, skipping: %s", match_url, e
                )
                continue

            all_records.extend(match_records)
    return pd.DataFrame(all_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = backfill_seasons(args.start_season, args.end_season)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("wrote %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
