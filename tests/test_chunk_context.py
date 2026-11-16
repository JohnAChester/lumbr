"""The three #4 additions to chunking (docs/SCHEMA.md): context carry for conversational sources,
Gmail quote stubs, and repeating a list field's label when its items continue into the next chunk."""

import re

import pytest

from lumbr import chunk
from test_chunk import DOCS, _norm

CONVERSATIONAL = ["basic_slack", "basic_gmail", "gmail_quoted_reply", "gmail_message_markers",
                  "basic_fireflies", "long_transcript"]
OTHER = ["basic_confluence", "basic_jira", "basic_linear", "list_fields_jira", "basic_github"]


@pytest.fixture(scope="module")
def passages():
    return {name: list(chunk.chunk_docs([doc])) for name, doc in DOCS.items()}


def _carries(ps):
    """(previous body, carried text) for every chunk after the first."""
    out = []
    for prev, cur in zip(ps, ps[1:]):
        rest = chunk.drop_carry(prev.body, cur.body)
        out.append((prev.body, cur.body[: len(cur.body) - len(rest)].rstrip("\n")))
    return out


# ---------------------------------------------------------------- context carry

def test_conversational_chunks_start_with_the_end_of_the_previous_chunk(passages):
    pairs = [c for name in CONVERSATIONAL for c in _carries(passages[name])]
    assert pairs
    with_carry = [(prev, carried) for prev, carried in pairs if carried]
    assert len(with_carry) >= 0.8 * len(pairs)
    for prev, carried in with_carry:
        assert prev.endswith(carried)


def test_carry_is_at_most_carry_max_tokens(passages):
    for name in CONVERSATIONAL:
        for _, carried in _carries(passages[name]):
            if carried:
                assert chunk.count([carried])[0] <= chunk.CARRY_MAX, name


def test_carry_starts_at_a_turn_line_or_sentence_boundary(passages):
    for name in CONVERSATIONAL:
        for prev, carried in _carries(passages[name]):
            if carried:
                before = prev[: len(prev) - len(carried)]
                assert before == "" or re.search(r"(\n[ \t]*|[.!?]\s+)$", before), (name, carried[:60])


def test_carry_has_no_leading_blank_lines(passages):
    for name in CONVERSATIONAL:
        for _, carried in _carries(passages[name]):
            assert carried == carried.lstrip(), (name, carried[:40])


def test_no_carry_for_other_sources(passages):
    for name in OTHER:
        for _, carried in _carries(passages[name]):
            assert carried == "", name


def test_drop_carry_removes_all_repeated_text(passages):
    """First body + every later body without its carry = the document's text once (small-to-big merge)."""
    for name in CONVERSATIONAL:
        ps = passages[name]
        once = ps[0].body + "".join(chunk.drop_carry(a.body, b.body) for a, b in zip(ps, ps[1:]))
        rendered = _norm("".join(chunk.render_units(DOCS[name], DOCS[name]["_source_type"])))
        assert len(_norm(once)) <= 1.02 * len(rendered), name


def test_drop_carry_leaves_unrelated_bodies_alone():
    assert chunk.drop_carry("alpha\nbeta", "gamma\ndelta") == "gamma\ndelta"


# ---------------------------------------------------------------- Gmail quote stubs

def test_quote_intro_becomes_a_stub():
    email = ("From: Jonas <j@x.com>\nSounds good, Thursday works.\n\n"
             "On Wed, Nov 18, 2026 at 02:14 AM, soojin_lee@redwood.com wrote:\n> Can we do Thursday?\n> ...")
    out = chunk.strip_quoted_reply(email)
    assert out.endswith("[quoted reply removed: On Wed, Nov 18, 2026 at 02:14 AM, soojin_lee@redwood.com wrote]")
    assert "Can we do Thursday" not in out


def test_original_message_and_bare_quotes_become_one_stub():
    assert chunk.strip_quoted_reply("Yes.\n-----Original Message-----\nFrom: a\nold") == "Yes.\n[quoted reply removed]"
    assert chunk.strip_quoted_reply("Yes.\n> old 1\n> old 2\nThanks") == "Yes.\n[quoted reply removed]\nThanks"


def test_fixture_reply_keeps_its_stub(passages):
    bodies = "\n".join(p.body for p in passages["gmail_quoted_reply"])
    assert "[quoted reply removed: On Wed, Nov 18, 2026 at 02:14 AM, soojin_lee@redwood.com wrote]" in bodies


# ---------------------------------------------------------------- list label continuation

def test_chunks_never_start_with_an_unlabelled_list_item(passages):
    for name in OTHER:
        for p in passages[name]:
            assert not chunk.drop_carry("", p.body).startswith("- "), (name, p.body[:60])


def test_continued_label_names_the_field(passages):
    starts = [p.body.split("\n", 2) for p in passages["list_fields_jira"][1:]]
    continued = [s for s in starts if s[0].endswith("(continued):")]
    assert continued
    for label, first_item, *_ in continued:
        assert re.fullmatch(r"[A-Z][\w /&-]* \(continued\):", label)
        assert first_item.startswith("- ")


# ---------------------------------------------------------------- transcript speaker formats

TRANSCRIPT_LINES = [
    "[00:15] Jordan Park: cool, quick agenda",           # already handled
    "09:00 Lena: Ok folks, thanks for joining",
    "00:00:02 Elena: Hey everyone",
    "00:00 - Olivia Chen: hey everyone",
    "00:00:12 — Ava: okay everyone",                 # em dash
    "05:41 - 06:25 Aisha: Ok good.",
    "00:00 Speaker 1 (Jasmine): hey everyone",
    "00:00 Maya (Redwood AE): Hey everyone",
    "[00:00] [00:05] Sam Lee: two timestamps",
    "[S00:18] Mark (Sentinel CISO): Thanks Ari.",
]


@pytest.mark.parametrize("line", TRANSCRIPT_LINES)
def test_transcript_speaker_formats_start_a_turn(line):
    text = "Header: Weekly sync\n" + line + "\ncontinued on a second line\n" + line
    turns = chunk.split_speaker_lines(text, chunk.TRANSCRIPT_SPEAKER)
    assert turns == ["Header: Weekly sync", line + "\ncontinued on a second line", line]


def test_fireflies_fixture_with_bare_timestamps_splits_into_turns():
    units = chunk.render_units(DOCS["basic_fireflies"], "fireflies")
    turns = [u for u in units if re.match(r"\d{2}:\d{2} ", u)]
    assert len(turns) > 10


def test_slack_log_lines_stay_inside_their_message():
    doc = {"dataset_doc_uuid": "dsid_" + "0" * 32, "_source_type": "slack", "content_field_names": ["text"],
           "text": "ops-bot: deploy log\n11:02:11 runtime(us-west): reject queue_full\n09:11:45 deploy-trigger: start"}
    assert len(chunk.render_units(doc, "slack")) == 1
