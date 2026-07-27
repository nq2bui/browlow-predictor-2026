from brownlow.names import normalize_player_name


def test_normalize_afltables_style_name():
    assert normalize_player_name("Baker, Liam") == "L. Baker"
    assert normalize_player_name("Bolton, Shai") == "S. Bolton"


def test_normalize_footywire_style_name():
    assert normalize_player_name("O Florent") == "O. Florent"
    assert normalize_player_name("J Selwood") == "J. Selwood"


def test_normalize_multi_word_surname():
    assert normalize_player_name("Van Berlo, Jack") == "J. Van Berlo"
