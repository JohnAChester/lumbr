# Abstention Detector (first-sentence rule) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count an answer as abstained when its lead (first line or first sentence) is the canned line or an own-words refusal. The study's scoring and the demo site apply the same rule.

**Architecture:** `lumbr.generate` owns the rule: `REFUSAL_PATTERNS`, `ABSTAIN_RULE`, `_lead`, and `parse_answer`. The demo's `export_parse_cases.py` exports the patterns plus test cases computed by Python into `parse_cases.json`. The site's `citations.ts` reads its patterns from that file, so there is no hand-copied list, and the site's case test pins TypeScript to Python.

**Tech Stack:** Python 3.11 + pytest (lumbr); TypeScript + vitest (personal_site, Next.js).

**Spec:** `docs/superpowers/specs/2026-11-23-abstention-detector-design.md` (approved 2026-11-24, commit 36b4a40)

## Branches

| Task | Repo | Branch |
|---|---|---|
| 1–3 | lumbr | `john/14d-abstention` (exists, pushed; based on `john/14-answer-generator`, PR #58) |
| 4 | lumbr | `john/14e-abstention-demo`: new, from `origin/john/18-interface-skeleton` with `john/14d-abstention` merged in (checked: the merge is clean). The exporter, `parse_cases.json` and `tests/demo/` exist only on the #18 branch. |
| 5 | personal_site | `john/lumbr-demo` (draft PR #3) |

Don't push or open PRs until John says so at the end; the PR targets depend on whether #58 and #57 have merged by then. The untracked `content/projects/cigna-*` files in personal_site belong to John. Leave them alone.

## Global Constraints

- The rule (spec, "The rule"): abstains ⇔ the lead starts with the canned line (`_starts_with_line`, ≤2 inserted words) **or** the normalised lead matches a `REFUSAL_PATTERNS` entry. Citations play no part. A canned line outside the lead doesn't count.
- The lead: text up to the first line break, or up to the first `.`, `!` or `?` followed by whitespace, whichever comes first. It's taken after stripping surrounding whitespace and normalised with `_normalise` (lowercase, curly quotes made straight, whitespace collapsed).
- `REFUSAL_PATTERNS`, exactly these four, in this order:
  - `\bthere is no\b`
  - `\bno (record|mention|information|single source)\b`
  - `\bcan(not|'t) answer\b`
  - `\b(sources|documents) do(es)? not (contain|mention|specify|include|provide|describe)\b`
- Patterns use only ASCII, `\b`, groups and alternation, so Python `re` and JS `RegExp` agree.
- `ABSTAIN_RULE = 2`. New runs write `"abstain_rule": 2` to `meta.json`.
- No new metric. `answerable_abstained_gold_present` is the "refused with gold retrieved" number.
- The prompt stays at v4 (`answer.json` unchanged). No run is regenerated.
- Test questions are never read. Dev runs only.
- Runs: `.venv/Scripts/python.exe -m pytest` (lumbr); `npx vitest run` (personal_site). On Windows use the scratchpad, not `/tmp`, and set `PYTHONIOENCODING=utf-8` when printing answers.

## Review Focus

1. **Negative-fact answers.** "There is no rate limit on Dedicated [2]." is a correct answer, but its lead matches `\bthere is no\b`, so it counts as abstained, and the demo shows the canned line instead. The spec accepts this and leaves it to the hand check. Task 4 pins it as a parse case so Python and TS at least agree, and Task 3's flip list must make it easy to spot.
2. **Leading blank lines or spaces** before a refusal ("\n\nThere is no record …") must still abstain. Pinned in Task 1 and the Task 4 cases.
3. **A curly apostrophe** in "can’t answer" must match `can(not|'t) answer`. Pinned in Task 1 and Task 4.
4. **Python/JS regex drift.** A pattern that compiles differently in JS (non-ASCII, lookbehind, flags) would silently split the site from the study. Task 1 asserts that every pattern is ASCII. Task 5 compiles every exported pattern.
5. **A markdown answer whose first line is a heading or intro** with bullets after it. Without the line-break cut, a bullet's "there is no …" would leak into the lead (the qst_0367 false hit). Pinned in Task 1 and Task 4.

---

### Task 1: The rule in `lumbr.generate`

**Files:**
- Modify: `src/lumbr/generate.py` (the answer-parsing section, lines ~145–200: constants, `_lead`, `parse_answer` and its docstring; `save_answers` meta)
- Test: `tests/test_generate.py`

**Interfaces:**
- Produces:
  - `REFUSAL_PATTERNS: tuple[str, ...]`, the four pattern strings, uncompiled, in spec order.
  - `ABSTAIN_RULE: int = 2`
  - `_lead(text: str) -> str`, the raw (un-normalised) lead.
  - `_lead_abstains(text: str, line: str) -> bool`
  - `parse_answer(...)`, signature unchanged, `abstained` now from `_lead_abstains`.
  - `save_answers` writes `abstain_rule` to meta.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_generate.py`; `P` is the test file's existing prompt fixture or `generate.load_prompt()`, and `src(n)` is its existing sources helper)

```python
def test_lead_cuts_at_line_break_or_sentence_end():
    assert generate._lead("Here are the rules:\n* There is no cap [1].") == "Here are the rules:"
    assert generate._lead("First one. Second one.") == "First one."
    assert generate._lead("\n\n  Only sentence") == "Only sentence"
    assert generate._lead("Version 2.5 is out. Next.") == "Version 2.5 is out."

def test_refusal_patterns_are_the_spec_list_and_ascii():
    assert generate.REFUSAL_PATTERNS == (
        r"\bthere is no\b", r"\bno (record|mention|information|single source)\b",
        r"\bcan(not|'t) answer\b",
        r"\b(sources|documents) do(es)? not (contain|mention|specify|include|provide|describe)\b")
    assert all(p.isascii() for p in generate.REFUSAL_PATTERNS)

@pytest.mark.parametrize("text", [
    "Based on the provided sources, there is no record of a Teams integration. Source [3] covers Slack [3].",
    "There is no mention of FP8 thresholds in the sources [1].",
    "I cannot answer that from these sources.",
    "I can’t answer that from these sources.",
    "The sources do not specify the queue-depth threshold. The interleaving ratio is 0.6 [1].",
    "\n\nNo information about that is available [2].",
])
def test_own_words_refusal_in_lead_abstains_even_when_citing(text):
    assert generate.parse_answer(text, src(5), P)["abstained"] is True

@pytest.mark.parametrize("text", [
    "Here are the requirements:\n* There is no limit on retries [1].\n* Timeouts are 30 s [2].",
    "The limit is 10 MiB [2]. There is no second limit.",
    "The rollout date is May 4 [2]. " + "I'm sorry, but I can't find that information in our database.",
])
def test_refusal_outside_the_lead_does_not_abstain(text):
    assert generate.parse_answer(text, src(5), P)["abstained"] is False

def test_negative_fact_answer_counts_as_abstained_by_rule():
    # Known trade-off (spec; Review Focus 1): a correct "there is no ..." answer reads as a refusal.
    assert generate.parse_answer("There is no rate limit on Dedicated [2].", src(5), P)["abstained"] is True

def test_save_records_abstain_rule(tmp_path):  # reuse test_save_writes_answers_and_meta's setup
    ...  # save two records, then:
    assert meta["abstain_rule"] == 2
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generate.py -q -k "lead or refusal or negative_fact or abstain_rule"`
Expected: FAIL, with `AttributeError: module 'lumbr.generate' has no attribute '_lead'` / `REFUSAL_PATTERNS`, and a `KeyError: 'abstain_rule'`.

- [ ] **Step 3: Implement in `src/lumbr/generate.py`**

`_lead`: `re.split(r"\n|(?<=[.!?])\s", text.strip(), maxsplit=1)[0]`. This is the exact split notebook 17's A/B cell inlined, so Task 3's parity check holds. Compile the patterns once into a private `_REFUSAL = re.compile("|".join(REFUSAL_PATTERNS))`.
`_lead_abstains(text, line)` is `_starts_with_line(lead, line) or bool(_REFUSAL.search(_normalise(lead)))`.
`parse_answer` sets `abstained = _lead_abstains(text, prompt["abstain"])`. Its docstring now says: a refusal in the lead (canned line or own words) counts even when the answer cites; a canned line elsewhere is a partial answer.
In `save_answers`, add `"abstain_rule": ABSTAIN_RULE` to `full_meta`.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass. The existing abstention tests (`test_abstention_*`, `test_partial_answer_is_not_abstention`, `test_abstention_line_later_in_answer_is_not_abstention`) still pass unchanged. If one fails, its text has a refusal phrase in the lead: stop and rule on it in the ledger rather than editing the test.

- [ ] **Step 5: Commit**

```bash
git add src/lumbr/generate.py tests/test_generate.py
git commit -m "Abstention: decide by the lead, canned line or own-words refusal (abstain_rule 2) (#14)"
```

### Task 2: Notebook 17 uses the shipped rule, with a before/after rescore cell

**Files:**
- Modify: `notebooks/17_john_generator.ipynb` (cell 17, the Prompt A/B cell; a new code cell after Compare, cell 15)

**Interfaces:**
- Consumes: `generate.rescore_records`, `generate._starts_with_line`, `generate.ABSTAIN_RULE` (Task 1).

- [ ] **Step 1: Prompt A/B cell.** Delete the inlined `_REFUSAL`, `lead_abstains` and `rescored`. Use `generate.rescore_records(recs, prompt)`, which now applies rule 2. Update the cell's header comment to match. Keep the checks and the flip printout unchanged.
- [ ] **Step 2: New cell "Abstention rule: before vs after"**, placed right after the Compare cell (cell 15). For every `gen_{RETRIEVER}_p*_dev_*` run, print one row: run name, abstained under rule 1 (`_starts_with_line(answer, prompt["abstain"])`), abstained under rule 2 (`rescore_records`), and the count that flipped each way. Then list each answered → abstained flip as `question_id`, type, gold in sources, and the first 160 characters of the answer, for John's hand check. Comment: "Rule 1 = canned line first (before 2026-11-24); rule 2 = generate.ABSTAIN_RULE".
- [ ] **Step 3: Check the notebook still parses**, and that the edit touched only those two cells.

Run: `.venv/Scripts/python.exe -c "import json;json.load(open('notebooks/17_john_generator.ipynb',encoding='utf-8'))" && git diff --stat`
Expected: no error; one file changed.

Write the notebook with `json.dumps(..., indent=1, ensure_ascii=False)` plus a trailing newline. That's the file's existing format, so the diff stays limited to the two cells.

- [ ] **Step 4: Commit**

```bash
git add notebooks/17_john_generator.ipynb
git commit -m "Notebook 17: Prompt A/B uses the shipped abstention rule; before/after rescore cell (#14)"
```

### Task 3: Acceptance rescore and the hand-check list (no product code)

**Files:**
- Create (scratchpad, not committed): `<scratchpad>/accept_rescore.py`, `<scratchpad>/abstention_flips.md`
- Inputs: `<scratchpad>/runs/slim.jsonl` (the 11 dev runs; fields `run`, `question_id`, `answer`, `sources`, `abstained`) and `<scratchpad>/runs/gen_{r1,r5_r6}_p{4,6}_qwen3.5-9b_dev_2026-11-24/answers.jsonl`. Gold doc ids come from `data/drive/raw/questions.parquet`, dev ids only (assert every id is in `splits/dev_question_ids.txt`).

**Interfaces:**
- Consumes: `generate.rescore_records`, `generate.load_prompt` (Task 1).

- [ ] **Step 1: Rescore all 15 runs** with `rescore_records` and compare against each record's saved `abstained` (rule 1).
- [ ] **Step 2: Check against the spec**

Expected:
- 11 old runs: exactly **24** answered → abstained flips; **0** abstained → answered.
- 4 L4 runs: abstention decisions **identical** to notebook 17's inlined rule. That's Task 1's `_lead_abstains` against a copy of the old cell's `lead_abstains`, kept in the script.

If the 11-run count isn't 24, list the difference by question and run. Don't change the patterns. Rule in the ledger, and bring the difference to John with the hand-check list.

- [ ] **Step 3: Write `abstention_flips.md`.** One line per flip: run, question_id, question type, gold in sources (yes/no), and the lead (≤200 chars). Sort the "gold in sources = yes" flips first, then any lead that begins with "there is no" (Review Focus 1), since those are where a false hit would hide.
- [ ] **Step 4: Ledger the counts.** No commit; John hand-checks the file before merge (spec, Acceptance).

### Task 4: Export the patterns and new cases for the site

**Files:**
- Branch setup: `git switch -c john/14e-abstention-demo origin/john/18-interface-skeleton && git merge --no-edit john/14d-abstention`
- Modify: `demo/build/export_parse_cases.py`
- Regenerate: `demo/content/parse_cases.json`
- Test: `tests/demo/test_parse_cases.py`

**Interfaces:**
- Consumes: `generate.REFUSAL_PATTERNS`, `generate.ABSTAIN_RULE`, `generate.parse_answer` (Task 1).
- Produces: `parse_cases.json` gains top-level `"abstain_rule": 2` and `"refusal_patterns": [four strings]`. Each case keeps `{text, n_sources, citations, invalid_citations, abstained}`.

- [ ] **Step 1: Run the full suite after the merge**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `test_parse_cases_match_python` may fail. If it does, it's on `prompt_sha256` (#55 moved the prompt file) or on an `abstained` value. Every other test passes. If anything else fails, debug before going on.

- [ ] **Step 2: Write the failing tests** in `tests/demo/test_parse_cases.py`

```python
def test_exports_the_refusal_patterns():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    assert data["refusal_patterns"] == list(generate.REFUSAL_PATTERNS)
    assert data["abstain_rule"] == generate.ABSTAIN_RULE

def test_cases_cover_the_lead_rule():
    texts = {c["text"]: c["abstained"] for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    for t, want in epc.LEAD_CASES:
        assert texts[t] is want, t
```

- [ ] **Step 3: Run and watch them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/demo/test_parse_cases.py -q`
Expected: FAIL, with `KeyError: 'refusal_patterns'` and `AttributeError: ... has no attribute 'LEAD_CASES'`.

- [ ] **Step 4: Add the cases and patterns to `export_parse_cases.py`**

Add a module constant `LEAD_CASES: list[tuple[str, bool]]`: text, then the expected `abstained`. All of them use `n_sources` 5, and `texts()` appends them.
- `("Based on the provided sources, there is no record of a Teams integration. Source [3] describes the Slack bot [3].", True)`: an own-words refusal that cites
- `("Here are the requirements:\n* There is no limit on retries [1].\n* Timeouts are 30 s [2].", False)`: a markdown first line, then bullets
- `("The sources do not specify the queue-depth threshold. The interleaving ratio is 0.6 [1].", True)`: refuses one part, answers the other
- `("I can’t answer that from these sources [2].", True)`: curly apostrophe
- `("\n\nThere is no mention of that in the sources.", True)`: leading blank lines
- `("There is no rate limit on Dedicated [2].", True)`: a negative fact (Review Focus 1)
- `("The limit is 10 MiB [2]. I cannot answer the second part.", False)`: refusal after the lead

The spec's "partial answer that ends with the canned line" case already exists (`"The rollout date is May 4 [2]. " + abstain`). Keep it.
`build()` adds `"abstain_rule": generate.ABSTAIN_RULE, "refusal_patterns": list(generate.REFUSAL_PATTERNS)` beside `abstain`. Update the module docstring: the site reads the patterns from this file.

- [ ] **Step 5: Regenerate and run the tests**

Run: `.venv/Scripts/python.exe demo/build/export_parse_cases.py && .venv/Scripts/python.exe -m pytest -q`
Expected: `wrote …parse_cases.json`; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add demo/build/export_parse_cases.py demo/content/parse_cases.json tests/demo/test_parse_cases.py
git commit -m "Parse cases: export the refusal patterns and lead-rule cases for the demo (#14, #18)"
```

### Task 5: The site applies the same rule

**Files (personal_site, `john/lumbr-demo`):**
- Move: `tests/lumbr-demo/parse_cases.json` → `src/lib/lumbr-demo/parse_cases.json`, replaced by Task 4's regenerated file. It moves because runtime code now imports the patterns, and `src/` must not import from `tests/`.
- Modify: `src/lib/lumbr-demo/citations.ts`, `tests/lumbr-demo/citations.test.ts` (import path, new tests)

**Interfaces:**
- Consumes: `parse_cases.json` `refusal_patterns: string[]` (Task 4).
- Produces: `parseAnswer(text, nSources, abstainLine)`. The signature and return shape are unchanged, so `ask.ts` needs no change. When the lead refuses, the visitor sees the canned line, as today.

- [ ] **Step 1: Move the file and copy in the new export**

```bash
git mv tests/lumbr-demo/parse_cases.json src/lib/lumbr-demo/parse_cases.json
cp ../lumbr/demo/content/parse_cases.json src/lib/lumbr-demo/parse_cases.json
```

In `citations.test.ts`, import cases from `@/lib/lumbr-demo/parse_cases.json` and add:

```ts
it("compiles every exported refusal pattern as plain ASCII", () => {
  for (const p of cases.refusal_patterns) {
    expect(/^[\x00-\x7f]*$/.test(p)).toBe(true);
    expect(() => new RegExp(p)).not.toThrow();
  }
});
```

Fix the header comments in both files so they name the new path.

- [ ] **Step 2: Run and watch the new cases fail**

Run: `npx vitest run tests/lumbr-demo/citations.test.ts`
Expected: FAIL on the `LEAD_CASES` texts that should abstain, for example `Based on the provided sources, there is no record…`, where the result has `abstained: false`.

- [ ] **Step 3: Implement in `citations.ts`**

Import `{ refusal_patterns }` from `./parse_cases.json` and build one `RegExp(refusal_patterns.join("|"))`. Add `lead(text)`: `text.trim().split(/\n|(?<=[.!?])\s/, 1)[0]`. The `limit 1` keeps only the first piece, the same as Python's `maxsplit=1` followed by `[0]`. Abstained = `startsWithLine(lead, abstainLine) || REFUSAL.test(normalise(lead))`. Update the header comment: the rule is `abstain_rule` 2, and the patterns come from the exported file.

- [ ] **Step 4: Run all site tests and the type check**

Run: `npx vitest run && npx tsc --noEmit`
Expected: all pass, with no type errors. `resolveJsonModule` is already on, since the test imports JSON today.

- [ ] **Step 5: Commit**

```bash
git add -A src/lib/lumbr-demo tests/lumbr-demo
git commit -m "Lumbr demo: abstain on a refusal in the answer's lead, patterns from parse_cases.json (lumbr #14)"
```

`git add` is scoped to those two paths, so John's untracked `content/projects/cigna-*` files stay out.

---

## Finish

Give John:
- Task 3's `abstention_flips.md` path, plus the counts: 24/0 on the 11 runs, and parity on the 4 L4 runs.
- The rulings.
- The open question of PR targets: `john/14d-abstention` → `john/14-answer-generator` or `main`, depending on #58; `john/14e-abstention-demo` → `john/18-interface-skeleton`; personal_site commits land on draft PR #3.

Nothing merges before John's hand check (spec, Acceptance).
