# Lumbr interface skeleton (#18): design

Owner: John. Status: draft for approval. Depends on: #14 (generator, PR #54). Repos: `lumbr` (search service,
content build) and `personal_site` branch `john/lumbr-demo` (the demo UI and its API routes).

## Intent

The "First Day at Redwood" demo plays a full round today, but only on fake data. #18 makes it real: a dev question
typed into Lumbr goes through **real retrieval** and **the study's generator**, and the answer comes back with
clickable `[n]` citations. The demo must run the same system the study measures, so context building and the
prompt come from the study's own code, not a copy (the same "certainty and reproducibility" rule as #14).

## Goals

- A small Python search service over the **1K-doc v0 sample** with R2 (`BAAI/bge-base-en-v1.5`), returning the
  top 5 hits, their small-to-big sources and the finished chat messages, built by `lumbr.generate`.
- The site calls the LLM with those messages through its existing spend/deadline/fallback code, then parses the
  answer with a TypeScript port of `parse_answer` that is pinned to the Python behaviour by shared test cases.
- The round's colleague questions are real dev questions answerable from the sample.
- **Done when:** locally (`npm run dev` + the service running locally + a real hosted Qwen2.5-7B-Instruct), a dev
  question typed into Lumbr returns a cited answer whose `[n]` popovers show the right sources, and an "info not
  found" question returns the canned line.

## Non-goals

- The full ~770K-passage index, threshold tuning, choosing the served retriever (B5, after #22).
- Hybrid/reranker serving (B6), cached safety-net answers and going live (site Task 10).
- Hand-polished colleague DMs for all 100 dev questions (B4 proper).

## Anti-goals

- The demo's context, prompt or abstention rule differing from the study's while claiming to be the same system.
- Any test question in the demo bundle or used to pick a threshold.
- Model or index files committed to either repo.

## Findings that shape it

- **11 of 100 dev questions** have all their gold documents in the v0 sample (8 basic, 1 project_related,
  1 conflicting_info, 1 miscellaneous). **4 more** are `info_not_found`. 15 questions; a round uses 6. (Two `high_level`
  dev questions also have no gold docs, but they need the whole corpus, so they're left out.)
- The site's current generator (`server/llm.ts`) has its own prompt citing `[doc_id]`, sends only 5 bare hits, uses
  `max_tokens` 400 and detects abstention with `startsWith`. All four change.
- R2 queries are encoded as `query_prefix + q` with **no space** after
  `"Represent this sentence for searching relevant passages:"`, normalised (`index_dense.R2`). Parity must match that.

## Approaches

- **A. Python builds context + messages, the site calls the LLM (chosen by John).** Exact study code for context
  and prompt; the site's tested safety code stays; only `parse_answer` is ported, pinned by shared cases.
- **B. Python does everything, including the LLM call.** Exact parsing too, but moves the LLM key, spend counter and
  deadline into Python (rewriting tested code), and every answer waits on the Python function's cold start.
- *Dropped:* all in TypeScript (copying `build_sources` and the prompt fill): the drift #14 was built to avoid.
- *Dropped:* the Python service inside `personal_site`: the study code would be copied or pip-installed from GitHub
  per build, and the site would inherit the service's size and memory limits.

## Decisions

Decided with John:

1. **Two repos.** Search service in `lumbr/demo/backend/` (own Vercel project, imports `lumbr`); UI and LLM calls
   stay in `personal_site`.
2. **v0 sample first** to show a working demo; the full index comes with B5.
3. **Approach A.**

Claude's calls (correct any of these):

4. **Contract change: `/retrieve` returns sources and messages.** Request
   `{query, k=5, context_ids=[], history=[]}`; response
   `{retriever, top_score, passed, prompt_sha256, hit_ids, sources: [Source], messages: [{role, content}]}`.
   `Source = Passage & {n, passage_ids, window_ids}`: the `Passage` fields come from the source's best-ranked hit,
   except `body` and `header`, which are the merged small-to-big window the model saw. So a citation popover shows
   exactly what the model read. `/passages` and `/health` stay; `/health` also reports `prompt_sha256`.
5. **Hits for a turn.** `passed` → this query's top-5 hits, then earlier `context_ids` not already present, capped at
   8 hits (the site's `MAX_CONTEXT_PASSAGES`). Not passed but `context_ids` given → earlier hits only. Neither →
   `passed: false`, no sources, no messages, and the site returns the canned line without an LLM call (unchanged).
   Sources are then `generate.build_sources(hits, by_id, neighbours=1)`.
6. **Follow-ups.** Messages = `build_messages(question, sources, prompt)` with prior turns inserted as
   user/assistant pairs between the system message and the final user message. The system prompt is unchanged.
   `[n]` markers are **stripped from earlier answers** first, since numbering restarts every turn. First-turn
   messages are byte-identical to the study's.
7. **Parsing in TypeScript.** `parseAnswer(text, nSources) -> {citations, invalidCitations, abstained}` ports
   `_CITE` (incl. `[n, m]`, excluding `[n](`) and `_starts_with_line` (≤ 2 inserted words, curly quotes, case,
   whitespace). A script in `lumbr` exports `parse_cases.json` (inputs + Python outputs + abstain line +
   `prompt_sha256`); the site commits it and tests `parseAnswer` against every case. Invalid citations are removed
   from the shown text; valid `[n]` keep their numbers and link to source n; `[1, 3]` is shown as `[1][3]` so each
   number becomes a chip.
8. **LLM settings match the study:** temperature 0, `max_tokens` 512, `Qwen2.5-7B-Instruct` on an OpenAI-compatible
   pay-per-token host (provider is John's pick; quantised hosting is noted later in the About panel).
9. **Index and model.** Encode the sample's passages with sentence-transformers (normalised); store **float32**
   numpy (sample is ~5K passages, ~15 MB) behind the same `Index` interface the int8 version will implement in B5.
   Query encoder: ONNX export of bge-base run with onnxruntime (torch is too large for the function), with a parity
   test against sentence-transformers (cosine ≥ 0.999 on 20 queries, same top-5 on all).
10. **Passage store:** SQLite with every passage of the sample docs (neighbours must be present), keyed by
    `passage_id`, indexed by `(doc_id, chunk_idx)`.
11. **Threshold (provisional):** the lowest top score among the 11 answerable questions minus 0.02, saved in
    `artifacts/config.json`. Only dev questions are read. Real tuning with an off-topic set is B5.
12. **Content bundle:** `demo/build/build_questions.py` writes `questions.json` for the 15 questions, failing loudly
    if the dev split file is missing. Colleague DMs come from `demo/content/colleague_dms.csv`, drafted by Claude
    for these 15 and reviewed by John. `cached: null`.
13. **Artifacts** (index, store, ONNX model, config) live in a private HF Hub repo and are fetched at build; never
    committed. For local runs they sit in `demo/backend/artifacts/` (git-ignored).
14. **Importing the study code without its heavy deps.** `lumbr.generate` imports only the standard library
    (checked: ~70 ms), but the `lumbr` package declares pandas, datasets and transformers. The service puts
    `src/` on its path (or installs `lumbr` with `--no-deps`) at the same commit, never a copy kept by hand. A test
    asserts that importing the service does not import pandas or torch.
15. **Deploy is the last step and optional for "done".** Deploying the service to Vercel needs John's Vercel and HF
    accounts; it is done if they're ready, with memory and cold-start measured.

## Pieces

| piece | repo / where | notes |
|---|---|---|
| build artifacts: embed sample, ONNX export, SQLite store, threshold | `lumbr/demo/build/` | needs `processed/sample_passages.jsonl` from Drive |
| service: encoder, index, store, `/retrieve` with sources + messages | `lumbr/demo/backend/` | FastAPI; tests with a hash encoder + fixtures |
| `parse_cases.json` export | `lumbr/demo/build/` | from `generate.parse_answer` |
| `questions.json` (15 questions) | `lumbr/demo/build/` → copied into the site | DMs reviewed by John |
| retrieval client + types for the new contract | `personal_site` `server/retrieval.ts`, `types.ts` | fake client updated too |
| drop the site prompt; send backend messages; 512 tokens | `personal_site` `server/llm.ts`, `server/ask.ts` | `LlmClient.chat(messages, signal)` |
| `parseAnswer` port + case tests; citations from sources | `personal_site` `citations.ts` | replaces `numberCitations` |
| end-to-end check, then optional deploy | both | |

## Left to the builder

FastAPI vs. a smaller framework, file layout inside `demo/`, ONNX export tooling, how the site maps a source to
its popover, the exact DM wording (John reviews), and the HF repo name.
