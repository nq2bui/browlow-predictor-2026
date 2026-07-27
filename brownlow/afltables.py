import re
from datetime import datetime

from bs4 import BeautifulSoup

_COLUMN_TO_FIELD = {
    "KI": "kicks",
    "HB": "handballs",
    "DI": "disposals",
    "MK": "marks",
    "GL": "goals",
    "BH": "behinds",
    "HO": "hitouts",
    "TK": "tackles",
    "CL": "clearances",
    "CP": "contested_possessions",
    "CM": "contested_marks",
    "GA": "goal_assists",
    "BR": "brownlow_votes",
}

SEASON_INDEX_URL_TEMPLATE = "https://afltables.com/afl/brownlow/brownlow{year}rbr.html"


def parse_match_header(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    summary_table = soup.find("table")
    text = summary_table.get_text(" ", strip=True)

    round_match = re.search(r"Round:\s*(\S+)", text)
    date_match = re.search(r"Date:\s*\w+,\s*(\d{1,2}-\w{3}-\d{4})", text)
    parsed_date = datetime.strptime(date_match.group(1), "%d-%b-%Y").strftime("%Y-%m-%d")

    team_links = [
        a.get_text(strip=True)
        for a in summary_table.find_all("a")
        if "teams/" in a.get("href", "")
    ]

    return {
        "round": round_match.group(1),
        "date": parsed_date,
        "home_team": team_links[0],
        "away_team": team_links[1],
    }


def parse_match_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table", class_="sortable"):
        header_th = table.find("th", colspan="25")
        team_name = header_th.get_text(strip=True).split(" Match Statistics")[0]
        header_cells = [th.get_text(strip=True) for th in table.find_all("tr")[1].find_all("th")]

        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            raw_row = dict(zip(header_cells, cells))

            parsed = {"team": team_name, "player": raw_row["Player"]}
            for abbr, field in _COLUMN_TO_FIELD.items():
                value = raw_row.get(abbr, "")
                parsed[field] = int(value) if value.isdigit() else 0
            rows.append(parsed)
    return rows


def list_season_match_urls(index_html: str) -> list[str]:
    hrefs = re.findall(r'href="(\.\./stats/games/[^"]+)"', index_html)
    seen = []
    for href in hrefs:
        path = href.split("stats/games/")[1]
        url = f"https://afltables.com/afl/stats/games/{path}"
        if url not in seen:
            seen.append(url)
    return seen
