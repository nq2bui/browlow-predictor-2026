import os

import pytest

from brownlow.odds import implied_probability, parse_brownlow_odds

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "sportsbet_brownlow_odds_sample.html"
)


def _load_fixture():
    with open(FIXTURE) as f:
        return f.read()


def test_parse_brownlow_odds_extracts_real_players_and_odds():
    odds = parse_brownlow_odds(_load_fixture())
    by_player = {o["player"]: o["decimal_odds"] for o in odds}

    # Names are normalized to the canonical "F. Surname" leaderboard join key.
    assert by_player["N. Daicos"] == pytest.approx(1.20)
    assert by_player["M. Bontempelli"] == pytest.approx(10.00)
    assert by_player["B. Smith"] == pytest.approx(21.00)
    assert by_player["J. Dawson"] == pytest.approx(23.00)
    assert by_player["I. Heeney"] == pytest.approx(46.00)
    assert by_player["K. Pickett"] == pytest.approx(51.00)


def test_parse_brownlow_odds_returns_normalized_dicts_in_document_order():
    odds = parse_brownlow_odds(_load_fixture())
    # The fixture holds exactly 6 complete outcome entries (a 7th name in the
    # real capture was truncated with no odds and must be dropped, not paired
    # against the wrong odds value).
    assert len(odds) == 6
    assert odds[0] == {"player": "N. Daicos", "decimal_odds": 1.20}
    for entry in odds:
        assert set(entry.keys()) == {"player", "decimal_odds"}
        assert isinstance(entry["decimal_odds"], float)


def test_parse_brownlow_odds_empty_html_returns_empty_list():
    assert parse_brownlow_odds("<html><body></body></html>") == []


def test_implied_probability_from_decimal_odds():
    assert implied_probability(1.20) == pytest.approx(83.3333, abs=1e-3)
    assert implied_probability(10.00) == pytest.approx(10.0)
    assert implied_probability(2.00) == pytest.approx(50.0)
