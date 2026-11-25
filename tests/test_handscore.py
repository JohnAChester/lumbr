"""Hand-scoring sample, sheets and agreement (#24): toy data, no Drive."""

import pytest

from lumbr import handscore as hs

# The test split's type counts (splits/split_manifest.json).
TEST_TYPES = {"basic": 140, "semantic": 100, "intra_document_reasoning": 32, "project_related": 32,
              "constrained": 24, "conflicting_info": 16, "completeness": 16, "miscellaneous": 16,
              "info_not_found": 16, "high_level": 8}


def _question_types():
    return {f"q_{t}_{i:03d}": t for t, n in TEST_TYPES.items() for i in range(n)}


# ---------------------------------------------------------------- sample

def test_allocate_test_split_sums_and_covers_every_type():
    alloc = hs.allocate(TEST_TYPES)
    assert sum(alloc.values()) == 50
    assert all(v >= 1 for v in alloc.values())
    assert alloc["basic"] == 18 and alloc["semantic"] == 12 and alloc["high_level"] == 1


def test_allocate_minimum_one_can_push_over_and_is_trimmed():
    alloc = hs.allocate({"big": 97, "a": 1, "b": 1, "c": 1}, n=4)
    assert alloc == {"big": 1, "a": 1, "b": 1, "c": 1}


def test_allocate_too_few_items():
    with pytest.raises(ValueError):
        hs.allocate({"a": 5, "b": 5, "c": 5}, n=2)


def test_draw_sample_shape():
    sample = hs.draw_sample(_question_types())
    assert [s["item"] for s in sample] == [f"h{i:02d}" for i in range(1, 51)]
    assert len({s["question_id"] for s in sample}) == 50
    per_run = {r: sum(s["run"] == r for s in sample) for r in hs.RUNS}
    assert set(per_run.values()) <= {7, 8} and sum(per_run.values()) == 50
    assert {s["question_type"] for s in sample} == set(TEST_TYPES)


def test_draw_sample_is_deterministic_and_seeded():
    qt = _question_types()
    assert hs.draw_sample(qt) == hs.draw_sample(dict(reversed(list(qt.items()))))
    assert hs.draw_sample(qt) != hs.draw_sample(qt, seed=hs.SEED + 1)


# ---------------------------------------------------------------- sheets

def test_strip_citations():
    assert hs.strip_citations("It is 5 [1][3]. Also x [1, 2].") == "It is 5. Also x."
    assert hs.strip_citations("See [docs](http://x) [2]") == "See [docs](http://x)"


def test_build_sheets_is_blind():
    sample = [{"item": "h01", "question_id": "q1", "run": "gen_r2_p7", "question_type": "basic"}]
    questions = {"q1": {"question": "Q?", "question_type": "basic", "gold_answer": "G",
                        "answer_facts": ["f one", "f two"]}}
    answers = {"gen_r2_p7": {"q1": {"answer": "A [1].", "abstained": False}}}
    sheet, key = hs.build_sheets(sample, questions, answers)
    assert sheet[0]["answer"] == "A."
    assert sheet[0]["gold_facts"] == "1. f one\n2. f two"
    assert "gen_r2_p7" not in str(sheet) and "q1" not in str(sheet)
    assert key == [{"item": "h01", "question_id": "q1", "run": "gen_r2_p7",
                    "question_type": "basic", "abstained": False}]
    assert set(sheet[0]) == set(hs.SHEET_COLUMNS)


def test_write_csv_round_trip_and_never_overwrites(tmp_path):
    rows = [{"item": "h01", "answer": "naïve, \"quoted\"\nline two"}]
    p = tmp_path / "s.csv"
    hs.write_csv(rows, p, ("item", "answer"))
    assert hs.read_csv(p) == rows
    with pytest.raises(FileExistsError):
        hs.write_csv(rows, p, ("item", "answer"))


# ---------------------------------------------------------------- filled sheets

def test_labels_from_sheet():
    rows = [{"item": "h01", "gold_facts": "1. a\n2. b\n3. c", "correct": "Yes", "facts": "1, 3"},
            {"item": "h02", "gold_facts": "1. a", "correct": "no", "facts": ""}]
    correct, facts = hs.labels_from_sheet(rows)
    assert correct == {"h01": True, "h02": False}
    assert facts == {("h01", 1): True, ("h01", 2): False, ("h01", 3): True, ("h02", 1): False}


@pytest.mark.parametrize("row", [
    {"item": "h07", "gold_facts": "1. a", "correct": "", "facts": ""},
    {"item": "h07", "gold_facts": "1. a", "correct": "maybe", "facts": ""},
    {"item": "h07", "gold_facts": "1. a", "correct": "yes", "facts": "2"},
])
def test_labels_from_sheet_rejects_bad_rows_naming_the_item(row):
    with pytest.raises(ValueError, match="h07"):
        hs.labels_from_sheet([row])


# ---------------------------------------------------------------- agreement

def test_cohen_kappa_known_value():
    # 2x2 table: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> kappa 0.4 (standard textbook example)
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert hs.cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edges():
    assert hs.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert hs.cohen_kappa([1, 0], [0, 1]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        hs.cohen_kappa([1], [1, 0])


def test_fleiss_kappa_known_value():
    # Fleiss (1971) style check: computed by hand. 4 items, 3 raters, 2 categories.
    ratings = [[1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1]]
    # P_i = 1, 1, 1/3, 1/3 -> P_bar = 2/3; p_yes = p_no = 1/2 -> P_e = 1/2; kappa = 1/3
    assert hs.fleiss_kappa(ratings) == pytest.approx(1 / 3)
    assert hs.fleiss_kappa([[1, 1], [1, 1]]) == 1.0


def test_fleiss_equals_cohen_shape_for_perfect_agreement():
    labels = {"a": {"h1": True, "h2": False}, "b": {"h1": True, "h2": False},
              "c": {"h1": True, "h2": False}}
    out = hs.agreement(labels)
    assert out["n"] == 2 and out["fleiss"] == 1.0
    assert out["pairs"] == {p: {"kappa": 1.0, "agree": 1.0} for p in ("a-b", "a-c", "b-c")}


def test_agreement_uses_shared_items_only_and_two_sets_have_no_fleiss():
    out = hs.agreement({"judge": {"h1": True, "h2": False, "h3": True},
                        "john": {"h1": True, "h2": False}})
    assert out["n"] == 2 and "fleiss" not in out
