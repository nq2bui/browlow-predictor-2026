from brownlow.names import join_key, normalize_player_name


def test_normalize_afltables_style_name():
    assert normalize_player_name("Baker, Liam") == "L. Baker"
    assert normalize_player_name("Bolton, Shai") == "S. Bolton"


def test_normalize_footywire_style_name():
    assert normalize_player_name("O Florent") == "O. Florent"
    assert normalize_player_name("J Selwood") == "J. Selwood"


def test_normalize_multi_word_surname():
    assert normalize_player_name("Van Berlo, Jack") == "J. Van Berlo"


def test_normalize_apostrophe_surname():
    assert normalize_player_name("O'Meara, Jaeger") == "J. O'Meara"
    assert normalize_player_name("J O'Meara") == "J. O'Meara"


def test_normalize_hyphenated_surname():
    assert normalize_player_name("Smith-Jones, Pat") == "P. Smith-Jones"
    assert normalize_player_name("P Smith-Jones") == "P. Smith-Jones"


def test_join_key_folds_apostrophe_differences():
    # afltables strips apostrophes in its own name text ("OSullivan, Finn"),
    # footywire keeps them ("Finn O'Sullivan") -- real, live-confirmed divergence.
    assert join_key(normalize_player_name("OSullivan, Finn")) == join_key(
        normalize_player_name("Finn O'Sullivan")
    )


def test_join_key_folds_case_differences():
    # afltables lowercases some prefixes footywire capitalizes, and vice versa --
    # real, live-confirmed divergence ("de Goey" vs "De Goey", "Macdonald" vs
    # "MacDonald").
    assert join_key(normalize_player_name("de Goey, Jordan")) == join_key(
        normalize_player_name("Jordan De Goey")
    )
    assert join_key(normalize_player_name("Macdonald, Connor")) == join_key(
        normalize_player_name("Connor MacDonald")
    )


def test_join_key_distinguishes_different_surnames():
    assert join_key(normalize_player_name("Smith, Bailey")) != join_key(
        normalize_player_name("Jones, Bailey")
    )
