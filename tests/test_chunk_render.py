"""Rendering edge cases that need no fixture file."""

import pytest

from lumbr import chunk


@pytest.mark.parametrize("source,field", [("slack", "message_count"), ("fireflies", "transcript")])
def test_non_text_content_fields_do_not_crash(source, field):
    """A few docs carry a number (or a dict) in a field the speaker splitter handles (crashed the #4 build)."""
    doc = {"content_field_names": ["text", field, "meta"], "text": "alice: hi\nbob: hello",
           field: 3, "meta": {"k": 1}}
    units = chunk.render_units(doc, source)
    assert any(u.endswith("3") for u in units)
    assert any("k: 1" in u for u in units)
