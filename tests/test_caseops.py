from luxbert.caseops import CaseOps

co = CaseOps()
M = co.marker


def test_basic_encode():
    assert co.encode("Hello") == f"{M}hello"
    assert co.encode("hello") == "hello"
    assert co.encode("HELLO") == f"{M}h{M}e{M}l{M}l{M}o"
    assert co.encode("iPhone") == f"i{M}phone"


def test_roundtrip_simple():
    for s in ["Hello World", "HELLO", "iPhone 15 PRO", "lëtzebuerg", "Lëtzebuerg"]:
        assert co.decode(co.encode(s)) == s, s


def test_luxembourgish_diacritics():
    # Ë / É / Ä / Ö / Ü should fold and restore exactly.
    for s in ["Ëimer", "Été", "Ärm", "Öffnung", "Über", "ZÄIT"]:
        assert co.roundtrip_ok(s), s


def test_marker_in_source_is_escaped():
    s = f"a{M}b{M}{M}C"
    enc = co.encode(s)
    assert co.decode(enc) == s


def test_digits_punct_whitespace_untouched():
    s = "12:34 — (a, b)\tnewline\n"
    assert co.encode(s) == s
    assert co.decode(s) == s


def test_empty_and_trailing_marker():
    assert co.encode("") == ""
    assert co.decode("") == ""
    # a lone marker decodes back to a literal marker
    assert co.decode(M) == M
