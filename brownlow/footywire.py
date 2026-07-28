import re

from bs4 import BeautifulSoup

MATCH_STATS_URL_TEMPLATE = "https://www.footywire.com/afl/footy/ft_match_statistics?mid={mid}&advv=Y"
SEASON_MATCH_LIST_URL_TEMPLATE = "https://www.footywire.com/afl/footy/ft_match_list?year={year}"


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
        si_index = headers.index("SI") - 1  # -1 because headers includes "Player" but tds after the player cell don't
        itc_index = headers.index("ITC") - 1

        for tr in table.find_all("tr", class_=["darkcolor", "lightcolor"]):
            tds = tr.find_all("td")
            player = tds[0].get_text(strip=True)
            values = [td.get_text(strip=True) for td in tds[1:]]
            rows.append({
                "team": team,
                "player": player,
                "score_involvements": int(values[si_index]) if values[si_index].replace(".", "").isdigit() else 0,
                "intercepts": int(values[itc_index]) if values[itc_index].replace(".", "").isdigit() else 0,
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
