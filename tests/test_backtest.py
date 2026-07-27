import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker
from brownlow.backtest import top20_hit_rate


def _row(player: str, match_id: str, votes: int, skill: int) -> dict:
    """One match row: every stat column is set to `skill` so a player's on-field
    quality is a single knob the ranker can learn from."""
    row = {col: skill for col in STAT_COLUMNS}
    row.update({"match_id": match_id, "brownlow_votes": votes, "player": player})
    return row


def _season_df_grinders_vs_spikes() -> pd.DataFrame:
    """A season purpose-built to expose the per-player-vs-per-row aggregation bug.

    top20_hit_rate MUST rank players by their SEASON TOTAL votes
    (df.groupby("player")["brownlow_votes"].sum()), not by their best single
    match. To force those two rankings apart we use two disjoint groups of
    players who never share a match:

    - 20 "GRIND" players: high skill, play 15 matches each and grind out just
      1 vote per match. Season total = 15, but every individual match row is a
      lowly 1-vote game.
    - 20 "SPIKE" players: lower skill, play exactly ONE match each and explode
      for 3 votes in it. Season total = 3, but each has a single 3-vote row.

    Consequences:
    - CORRECT ranking (sum per player, then take top 20): the 20 GRIND players
      (total 15) dominate the 20 SPIKE players (total 3). Actual top-20 = the
      grinders, and because grinders also carry the highest stats over the most
      matches the model's summed prediction recovers exactly that set, so the
      hit rate is 1.0.
    - BUGGY ranking (take the top-20 individual match ROWS by single-match
      votes, skipping the groupby/sum): the 3-vote SPIKE rows outrank every
      1-vote GRIND row, so the "actual top-20" becomes the 20 spikes while the
      model's top predicted rows are the high-skill grinder rows. Those sets are
      disjoint, driving the hit rate to ~0.0.

    So the correct implementation scores 1.0 here and the per-row bug scores
    ~0.0 -- the assertion below distinguishes them.
    """
    rows = []
    match_num = 0

    grinders = [f"GRIND{i:02d}" for i in range(20)]
    spikes = [f"SPIKE{i:02d}" for i in range(20)]

    # Grinder matches: all 20 grinders play together, 1 vote each, over 15 rounds.
    # A 0-vote filler gives each match a clear stat/vote ordering to learn from.
    for j in range(15):
        match_id = f"grind_m{match_num}"
        match_num += 1
        for g in grinders:
            rows.append(_row(g, match_id, votes=1, skill=80))
        rows.append(_row(f"grindfill{j}", match_id, votes=0, skill=5))

    # Spike matches: each spike gets its own match and a single 3-vote explosion,
    # padded with 0-vote fillers so the spike is the clear best player there.
    for i, s in enumerate(spikes):
        match_id = f"spike_m{match_num}"
        match_num += 1
        rows.append(_row(s, match_id, votes=3, skill=40))
        rows.append(_row(f"spikefill{i}a", match_id, votes=0, skill=5))
        rows.append(_row(f"spikefill{i}b", match_id, votes=0, skill=5))

    return pd.DataFrame(rows)


def test_top20_hit_rate_ranks_by_season_total_not_single_match():
    """Guards against ranking by single-match rows instead of per-player season
    totals -- the exact bug top20_hit_rate exists to prevent.

    See _season_df_grinders_vs_spikes for the data shape: a correct
    (group-by-player, sum-votes) implementation scores exactly 1.0, whereas a
    version that takes the raw top-20 match rows (no per-player summing) scores
    ~0.0 because the grinders' season-leading totals never appear as top single
    rows. The `== 1.0` assertion below therefore fails against that regression.
    """
    df = _season_df_grinders_vs_spikes()
    model = train_ranker(df)
    hit_rate = top20_hit_rate(model, df)

    assert 0.0 <= hit_rate <= 1.0  # sanity: it is a fraction
    # The 20 grinders lead the season on summed votes AND on summed predictions,
    # so a correct per-player implementation recovers all of them. A per-row
    # implementation would score ~0.0 here (disjoint grinder/spike sets).
    assert hit_rate == 1.0
