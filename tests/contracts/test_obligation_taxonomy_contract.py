"""One obligation vocabulary, two languages.

The Go proxy derives obligations from the validated request; the V3 service
measures records against them. A closed set written twice is a divergence
waiting to happen: a kind added on one side is a kind the other silently
treats as unknown, and an id computed differently is evidence that binds to
nothing.

This test parses both sources and fails when they disagree about:

  - which kinds exist,
  - what floor each kind demands,
  - which kinds take their floor from a baseline,
  - which kinds nothing can satisfy,
  - the exact id string a (kind, subject) pair produces.

Source parsing only. Nothing here imports the proxy or runs a service.
"""

import hashlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GO_SRC = REPO / "proxy" / "obligation_kinds.go"
GO_TEST = REPO / "proxy" / "obligation_kinds_test.go"
PY_SRC = REPO / "v3-service" / "obligations.py"

sys.path.insert(0, str(REPO / "v3-service"))


@pytest.fixture(scope="module")
def taxonomy():
    import obligations  # noqa: E402
    return obligations


def _go_text():
    assert GO_SRC.exists(), f"{GO_SRC} is missing"
    return GO_SRC.read_text()


def _go_const_kinds():
    """The kind string literals from the Go const block."""
    text = _go_text()
    block = text[text.index("const ("):text.index("// obligationKinds is the closed set")]
    return set(re.findall(r'=\s*"([a-z_]+)"', block))


def _go_ordered_kinds():
    """The declaration order Go publishes as the closed set."""
    text = _go_text()
    start = text.index("var obligationKinds = []string{")
    block = text[start:text.index("}", start)]
    return re.findall(r"\bObligation([A-Za-z]+),", block)


def _go_map(name):
    text = _go_text()
    start = text.index(f"var {name} = map[string]")
    block = text[start:text.index("\n}", start)]
    return dict(re.findall(r"\bObligation([A-Za-z]+):\s*\"([a-z]+)\"", block))


def _go_bool_set(name):
    text = _go_text()
    start = text.index(f"var {name} = map[string]bool{{")
    block = text[start:text.index("\n}", start)]
    return set(re.findall(r"\bObligation([A-Za-z]+):\s*true", block))


def _go_symbol_to_value():
    """ObligationArtifactExists -> "artifact_exists", read from the const block."""
    text = _go_text()
    block = text[text.index("const ("):text.index("// obligationKinds is the closed set")]
    return dict(re.findall(r"\b(Obligation[A-Za-z]+)\s*=\s*\"([a-z_]+)\"", block))


def test_both_sources_exist():
    assert GO_SRC.exists()
    assert PY_SRC.exists()


def test_the_kind_sets_are_identical(taxonomy):
    go_kinds = _go_const_kinds()
    assert go_kinds == set(taxonomy.KINDS), (
        f"only in Go: {sorted(go_kinds - set(taxonomy.KINDS))}; "
        f"only in Python: {sorted(set(taxonomy.KINDS) - go_kinds)}")


def test_the_closed_set_is_published_in_the_same_order(taxonomy):
    symbols = _go_symbol_to_value()
    ordered = [symbols["Obligation" + s] for s in _go_ordered_kinds()]
    assert ordered == list(taxonomy.KINDS), (
        "the two sides publish the closed set in different orders, so a "
        "reader comparing them positionally would pair the wrong kinds")


def test_the_fixed_floors_agree(taxonomy):
    symbols = _go_symbol_to_value()
    go_floors = {symbols["Obligation" + s]: v
                 for s, v in _go_map("obligationKindRequiredStrength").items()}
    assert go_floors == dict(taxonomy._KIND_REQUIRED_STRENGTH)


def test_the_dynamic_and_unsatisfiable_kinds_agree(taxonomy):
    symbols = _go_symbol_to_value()
    go_dynamic = {symbols["Obligation" + s]
                  for s in _go_bool_set("obligationDynamicStrengthKinds")}
    go_never = {symbols["Obligation" + s]
                for s in _go_bool_set("obligationUnsatisfiableKinds")}
    assert go_dynamic == set(taxonomy._DYNAMIC_STRENGTH_KINDS)
    assert go_never == set(taxonomy._UNSATISFIABLE_KINDS)


def test_every_kind_is_classified_exactly_once(taxonomy):
    """Fixed, baseline-floored, or unsatisfiable. A kind in none of the three
    has no answer about what closes it; a kind in two has two."""
    fixed = set(taxonomy._KIND_REQUIRED_STRENGTH)
    dynamic = set(taxonomy._DYNAMIC_STRENGTH_KINDS)
    never = set(taxonomy._UNSATISFIABLE_KINDS)
    assert fixed | dynamic | never == set(taxonomy.KINDS)
    assert not (fixed & dynamic) and not (fixed & never) and not (dynamic & never)


def test_the_id_derivation_agrees_with_the_pinned_go_vectors(taxonomy):
    """The Go test pins exact id strings. They are recomputed here from the
    Python derivation, so a change to either side breaks this."""
    go_test = GO_TEST.read_text()
    vectors = re.findall(
        r'\{Obligation([A-Za-z]+), "([^"]+)",\s*\n?\s*"([a-z_]+:[0-9a-f]{32})"',
        go_test)
    assert len(vectors) >= 3, f"expected pinned Go vectors, found {vectors}"
    symbols = _go_symbol_to_value()
    for symbol, subject, pinned in vectors:
        kind = symbols["Obligation" + symbol]
        assert taxonomy.obligation_id(kind, subject) == pinned, (
            f"{kind}/{subject!r}: Go pins {pinned}, Python computes "
            f"{taxonomy.obligation_id(kind, subject)}")


def test_the_id_is_a_kind_scoped_sha256_prefix(taxonomy):
    """Stated independently of both implementations, so a shared mistake in
    the two derivations is still caught."""
    subject = "solve.py"
    want = "artifact_exists:" + hashlib.sha256(subject.encode()).hexdigest()[:32]
    assert taxonomy.obligation_id(taxonomy.KIND_ARTIFACT_EXISTS, subject) == want


def test_neither_side_names_a_hidden_evaluator():
    """The benchmark's own grader has no representation in either vocabulary.
    Absent by construction, so production cannot name it even by mistake."""
    for path in (GO_SRC, PY_SRC):
        body = path.read_text().lower()
        for banned in ("holdout", "hidden_evaluator", "benchmark_grader",
                       "reference_answer"):
            assert banned not in body, f"{banned} appears in {path.name}"


def test_the_subject_never_travels_in_an_id(taxonomy):
    """A declared command is a subject. The id is a hash of it and cannot
    contain it, so an id in a log is never a content leak."""
    secret = "pytest --token=hunter2 -q"
    oid = taxonomy.obligation_id(taxonomy.KIND_DECLARED_COMMAND, secret)
    assert secret not in oid and "hunter2" not in oid
