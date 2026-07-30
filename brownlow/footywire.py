import re

from bs4 import BeautifulSoup

from brownlow.names import normalize_player_name

MATCH_STATS_URL_TEMPLATE = "https://www.footywire.com/afl/footy/ft_match_statistics?mid={mid}&advv=Y"
SEASON_MATCH_LIST_URL_TEMPLATE = "https://www.footywire.com/afl/footy/ft_match_list?year={year}"
# footywire's per-team, per-season player-list ("team players") page. Confirmed
# to support historical years back to at least 2012 and up to the current
# roster. Slugs live in brownlow.teams.FOOTYWIRE_TEAM_SLUGS.
TEAM_ROSTER_URL_TEMPLATE = "https://www.footywire.com/afl/footy/tp-{slug}?year={year}"


def parse_team_roster(html: str) -> list[dict]:
    """Parse a footywire team-roster page into ``[{"player", "position"}, ...]``.

    Returns one dict per player row, with ``player`` normalized to the canonical
    "F. Surname" join key (footywire lists names as "Surname, First", the same
    convention afltables uses, so ``normalize_player_name`` handles it directly)
    and ``position`` the raw, whitespace-stripped position string from the LAST
    ``<td class="data">`` cell of the row (e.g. "Forward", "Defender",
    "Midfield", "Ruck", or a combo like "MidfieldForward").

    Parsed with lxml specifically: footywire's roster table emits a stray extra
    ``</tr>`` after every row, which Python's built-in html.parser mis-nests --
    it closes the table after the first row and drops every subsequent player.
    lxml recovers the intended structure and returns all rows. A page missing
    the ``team-players-div`` container (an error page, an unexpected layout)
    yields an empty list rather than raising, matching the pipeline's
    graceful-degradation convention.
    """
    soup = BeautifulSoup(html, "lxml")
    div = soup.find("div", id="team-players-div")
    if div is None:
        return []

    players = []
    for tr in div.find_all("tr", class_=["darkcolor", "lightcolor"]):
        name_link = tr.find("a")
        data_cells = tr.find_all("td", class_="data")
        # A genuine player row has a name link and several data cells (the last
        # being Position). Skip anything that doesn't (defensive against stray
        # or malformed rows), rather than indexing into an empty list.
        if name_link is None or not data_cells:
            continue
        # The name cell may carry an extra <span class="playerflag" title=
        # "Rookie">R</span> after the link; taking the <a>'s own text (not the
        # whole cell) keeps the "R" flag out of the parsed name.
        raw_name = name_link.get_text(strip=True)
        position = data_cells[-1].get_text(strip=True)
        players.append({
            "player": normalize_player_name(raw_name),
            "position": position,
        })
    return players


def parse_advanced_stats_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    teams = []
    for title_td in soup.find_all("td", class_="innertbtitle"):
        text = title_td.get_text(strip=True)
        if "Match Statistics" in text:
            teams.append(text.split(" Match Statistics")[0])

    candidate_tables = [t for t in soup.find_all("table") if t.find("span", attrs={"title": "Player"})]
    stats_tables = [
        t for t in candidate_tables
        if not any(t is not other and other in t.find_all("table") for other in candidate_tables)
    ]

    rows = []
    for team, table in zip(teams, stats_tables):
        headers = [s.get_text(strip=True) for s in table.find_all("span", class_="sortByAjaxLink")]
        # -1 because headers includes "Player" but the tds after the player
        # cell don't. footywire's older (2012-era) advanced-stats pages track a
        # SMALLER column set that lacks "SI" (Score Involvements) and "ITC"
        # (Intercepts) entirely -- those were added by footywire sometime after
        # 2012. When a column is absent for this page/table, default its stat to
        # 0 for every player rather than crashing on headers.index(...), matching
        # how the pipeline already treats data it can't find.
        # -1 on every index for the same reason as SI/ITC above: headers
        # includes "Player" (index 0) but the tds after the player cell (values)
        # don't, so a header at index N maps to values index N-1. All of these
        # newer advanced-stats columns are absent on footywire's pre-2015 pages
        # (the smaller 11-column set), so each is guarded with "in headers" and
        # defaults to 0 for every player when the column doesn't exist.
        def col_index(name):
            return headers.index(name) - 1 if name in headers else None

        si_index = col_index("SI")
        itc_index = col_index("ITC")
        up_index = col_index("UP")
        ed_index = col_index("ED")
        de_index = col_index("DE%")
        mi5_index = col_index("MI5")
        onep_index = col_index("1%")
        ccl_index = col_index("CCL")
        mg_index = col_index("MG")
        t5_index = col_index("T5")
        # Highest column index this table actually needs to read; used to skip
        # short rows (e.g. the 2-cell "Unused Substitute" row). None-safe: only
        # present columns contribute. -1 when none are present (nothing to read).
        needed_indices = [
            i for i in (
                si_index, itc_index, up_index, ed_index, de_index, mi5_index,
                onep_index, ccl_index, mg_index, t5_index,
            ) if i is not None
        ]
        max_needed = max(needed_indices) if needed_indices else -1

        for tr in table.find_all("tr", class_=["darkcolor", "lightcolor"]):
            tds = tr.find_all("td")
            # footywire suffixes a substituted player's name with an arrow marker
            # directly appended with no separator: U+2197 (↗, subbed ON) or
            # U+2199 (↙, subbed OFF), e.g. "N Vlastuin↗". afltables carries no
            # such marker, so these names never matched on the (team, name) join
            # key and silently lost their SI/ITC. Strip the trailing arrow (and
            # any surrounding whitespace) before the name is used, so it
            # normalizes identically to afltables' plain name.
            player = tds[0].get_text(strip=True).rstrip("↗↙").strip()
            values = [td.get_text(strip=True) for td in tds[1:]]
            # A player named as an emergency/substitute who never took the field
            # gets a short "Unused Substitute" row: just the name and a single
            # colspan cell (2 <td>s), not the 18 a genuine player row has. Its
            # values list is too short to hold the SI/ITC columns, so indexing
            # it raised IndexError and crashed the whole backfill. Such a player
            # legitimately has no stats -- skip the row entirely, matching how
            # the pipeline already treats a player it finds no data for, rather
            # than recording them with fabricated zeroes.
            if len(values) <= max_needed:
                continue

            def stat(index):
                if index is None:
                    return 0
                return int(values[index]) if values[index].replace(".", "").isdigit() else 0

            def stat_float(index):
                # disposal_efficiency (DE%) is a percentage like "73.1", not a
                # whole integer, so it must parse with float() -- int("73.1")
                # would raise. The digit check ("73.1".replace(".","").isdigit()
                # -> True) is shared, but the cast is float, not int.
                if index is None:
                    return 0
                return float(values[index]) if values[index].replace(".", "").isdigit() else 0

            rows.append({
                "team": team,
                "player": player,
                "score_involvements": stat(si_index),
                "intercepts": stat(itc_index),
                "uncontested_possessions": stat(up_index),
                "effective_disposals": stat(ed_index),
                "disposal_efficiency": stat_float(de_index),
                "marks_inside_50": stat(mi5_index),
                "one_percenters": stat(onep_index),
                "centre_clearances": stat(ccl_index),
                "metres_gained": stat(mg_index),
                "tackles_inside_50": stat(t5_index),
            })
    return rows


def list_season_match_ids(match_list_html: str) -> list[dict]:
    soup = BeautifulSoup(match_list_html, "html.parser")
    matches = []
    for tr in soup.find_all("tr", class_=["darkcolor", "lightcolor"]):
        tds = tr.find_all("td")
        # Real footywire match-list pages reuse the darkcolor/lightcolor classes
        # on unrelated account-settings widgets ("Change Password"/"Update
        # Settings"), which have a single colspan cell rather than the date,
        # team-links, venue and result cells a genuine match row has. Skip any
        # row that lacks the cells this function needs (previously raised
        # IndexError on tds[1]/tds[3]).
        if len(tds) < 4:
            continue
        team_links = tds[1].find_all("a")
        if len(team_links) != 2:
            continue
        # The match-stats link is NOT at a fixed column index. On real
        # completed-match rows it lives in the result column (index 4), while
        # the attendance column (index 3) holds a plain number with no <a>.
        # Search the whole row for the link by what it actually IS -- an
        # ft_match_statistics?mid=... anchor -- rather than by position.
        mid_link = tr.find("a", href=re.compile(r"ft_match_statistics\?mid=\d+"))
        if mid_link is None:
            continue
        mid_match = re.search(r"mid=(\d+)", mid_link["href"])
        if mid_match is None:
            continue
        matches.append({
            "mid": int(mid_match.group(1)),
            "home_team": team_links[0].get_text(strip=True),
            "away_team": team_links[1].get_text(strip=True),
        })
    return matches
