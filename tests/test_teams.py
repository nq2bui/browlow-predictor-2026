from brownlow.teams import canonicalize_team_name, get_team_info


def test_canonicalize_maps_brisbane_alias():
    # footywire says "Brisbane"; afltables (the canonical spelling) says
    # "Brisbane Lions". Canonicalization maps the alias to the afltables form.
    assert canonicalize_team_name("Brisbane") == "Brisbane Lions"


def test_canonicalize_maps_gws_alias():
    # footywire says "GWS"; afltables says "Greater Western Sydney".
    assert canonicalize_team_name("GWS") == "Greater Western Sydney"


def test_canonicalize_is_noop_for_unaliased_name():
    # Any of the 16 identically-spelled clubs (and anything unrecognized) passes
    # through unchanged -- canonicalize is a safe no-op, never a lookup failure.
    assert canonicalize_team_name("Richmond") == "Richmond"


def test_canonicalize_passthrough_for_canonical_spelling():
    # Feeding the already-canonical afltables spelling back in is idempotent.
    assert canonicalize_team_name("Brisbane Lions") == "Brisbane Lions"
    assert canonicalize_team_name("Greater Western Sydney") == "Greater Western Sydney"


def test_canonicalized_alias_resolves_to_real_team_info():
    # The whole point: after canonicalizing footywire's alias, get_team_info
    # (keyed on afltables spelling) resolves to the real club, not the fallback.
    info = get_team_info(canonicalize_team_name("Brisbane"))
    assert info["code"] == "BRI"
