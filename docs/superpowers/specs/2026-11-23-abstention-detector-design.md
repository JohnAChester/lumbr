# Abstention detector: decide by the first sentence (#14 follow-up)

**Date:** 2026-11-23 · **Branch:** `john/14d-abstention` (from `john/14-answer-generator`) · **Site:** personal_site `john/lumbr-demo`

## Why

`parse_answer` counts an answer as abstained only when it **opens with** the canned line
("I'm sorry, but I can't find that information in our database.", allowing ≤2 extra words). Qwen3.5-9B
usually refuses in its own words and then points at related sources, for example "Based on the provided
sources, there is no record of …" followed by "Source [3] describes …". Those answers are scored as
answered. That raises the answered count and lowers the abstained count in `summarize`, and both
numbers feed #22's report and the demo, which should show the canned line when the model refuses.

## What the dev runs show

All 11 saved dev runs (1,100 answers) were checked against the gold docs. Test questions were not
read.

- **24 refusals are missed**, all in Qwen3.5 runs. Each one opens with "there is no …", "no
  mention/record/information of …", "I cannot answer …" or "the sources do not contain …".
  - In **12**, the gold doc was not retrieved. The refusal is correct, and the citations point at
    near-misses that the model labels "related but distinct".
  - In **12**, the gold doc was retrieved and cited, but the model refused anyway. It took the
    question's paraphrase too literally (qst_0194, 0252), or it refused one part of a two-part
    question (qst_0404).
- **87 answers open with the canned line and then cite sources.** Today these are correctly counted
  as abstentions.
- **Qwen2.5 sometimes ends a partial answer with the canned line**, for the one sub-question it
  can't answer (qst_0428). That is a partial answer, not an abstention.

## Approaches

| | Rule | Verdict |
|---|---|---|
| A | It refuses **and** cites no valid source (approved first) | **Rejected by the data.** Qwen3.5 cites while refusing, so none of the 24 misses would be caught, and the 87 abstentions that cite would flip to answered. |
| **B** | **The first sentence is the canned line or a refusal; citations are ignored** | **Chosen.** It catches the 24 and keeps the 87 as abstentions. A canned line later in the answer reads as a partial answer, which counts as answered. It is deterministic and ports to TypeScript. |
| C | Three labels (answered / partial / abstained), or an LLM decides | Deferred. #23's LLM judge grades correctness and can handle a "refused but had the facts" answer. The study's abstention count stays a cheap deterministic rule. |

## The rule (`abstain_rule` 2)

An answer abstains when its **lead** meets either of these conditions:
1. It starts with the canned line, as today (`_starts_with_line`, ≤2 inserted words), or
2. It matches one of `REFUSAL_PATTERNS`.

- **The lead** is the text up to the first line break, or up to the first `.`, `!` or `?` followed by
  whitespace, whichever comes first. It is normalised the way `_normalise` does it: lowercased, curly
  quotes made straight, whitespace collapsed. Cutting at the line break keeps a markdown answer's
  first line from running into its bullets. One false hit in the spot check (qst_0367) came from
  exactly that.
- **`REFUSAL_PATTERNS`** is a short list of lowercase regexes in `lumbr.generate`, drawn from the real
  dev misses:
  - `\bthere is no\b`
  - `\bno (record|mention|information|single source)\b`
  - `\bcan(not|'t) answer\b`
  - `\b(sources|documents) do(es)? not (contain|mention|specify|include|provide|describe)\b`
  - The patterns use only syntax that works the same in Python `re` and JavaScript `RegExp`: ASCII,
    `\b`, groups and alternation.
- **Citations play no part** in the abstention decision. They are still parsed as before.
- **A canned line anywhere other than the lead does not count.**

## Changes

**lumbr**
- `generate.py`:
  - Adds `REFUSAL_PATTERNS`, `ABSTAIN_RULE = 2` and `_lead(text)`.
  - `parse_answer` uses the rule above.
  - The docstring is updated to say that a refusal in the lead counts even when the answer cites.
- **Run metadata:** new runs record `abstain_rule` in `meta.json`. `rescore_records` already re-parses
  saved runs with the current rule, so older runs can be compared.
- **Metrics:** no new metric. `summarize`'s existing `answerable_abstained_gold_present` is the
  "refused although the gold doc was retrieved" number, and #22 reports it on its own line, separate
  from retrieval misses.
- **`demo/build/export_parse_cases.py`:**
  - Exports `refusal_patterns` alongside the cases.
  - Adds cases for:
    - an own-words refusal that also cites (abstains)
    - a markdown first line followed by bullets (doesn't abstain)
    - a partial answer that ends with the canned line (doesn't abstain)
    - a refusal of one part followed by an answer to the other (abstains)
- **Tests:** unit tests for `_lead` and the new cases in `test_generate.py`. `test_parse_cases.py`
  keeps the exported file in sync.
- **Notebook 17:** a short rescore cell showing abstained counts before and after the rule for every
  saved dev run.

**personal_site**
- `citations.ts`'s `parseAnswer` applies the same lead rule, using the `refusal_patterns` exported in
  `parse_cases.json` instead of a hand-copied list. The existing parse-case test pins it.
- **The demo's behaviour:** when the lead refuses, the visitor sees the canned line, as today.

## Acceptance

- Rescoring the 11 dev runs flips exactly the reviewed set to abstained. The expected set is the 24
  above, minus any false hits found in the hand check. No answer flips from abstained to answered.
- The same check runs on the four L4 dev runs from 2026-11-24 (`gen_r1_p4`, `gen_r1_p6`, `gen_r5_r6_p4`
  and `gen_r5_r6_p6`, each `_qwen3.5-9b_dev_2026-11-24`, on Drive). Their flips join the hand-checked list.
  The inlined rule in notebook 17's Prompt A/B cell already caught every v4 refusal in them, and the
  shipped rule must give the same abstention decisions as that cell on all four.
- John hand-checks the list of flips before merge.
- Python and TypeScript agree on every case in `parse_cases.json`.

## Not in scope

- **Changing the prompt** so the model answers paraphrased questions and answers the parts it can.
  The 12 gold-retrieved refusals suggested it, and it has since been tried: rounds 5 (2026-11-23) and 6
  (2026-11-24) both loosened rule 3, and neither was adopted. Each fixed a few literal refusals but
  answered more questions from look-alike documents (notebook 17's rounds table). `answer.json` stays at v4,
  so this detector has to handle v4's own-words refusals.
- **A "partial" label.**
- **Regenerating any run.** Rescoring is enough.

## Left to the builder

- Exact names of the helper and of the new tests.
- Whether `ABSTAIN_RULE` is also written by `rescore_records` callers in the notebook.
