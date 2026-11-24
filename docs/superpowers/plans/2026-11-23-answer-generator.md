# Answer Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lumbr's fixed answer generator (#14): top-5 passages + neighbours in, a cited answer or an abstention out, run over the 100 dev questions on the R1 (BM25) dev run.

**Architecture:** All logic that needs no GPU lives in `src/lumbr/generate.py` (context building, prompt, citation parsing, the generate loop with an injected chat function, saving), tested with pytest and fakes. The prompt is a JSON file in the package so the demo can copy it and verify its hash. Notebook 17 loads Qwen2.5-7B-Instruct in vLLM on an L4 and calls the module.

**Tech Stack:** Python 3.10+, pytest, vLLM (`<0.34`, notebook only), `Qwen/Qwen2.5-7B-Instruct`.

**Spec:** `docs/superpowers/specs/2026-11-23-answer-generator-design.md`

## Global Constraints

- Model `Qwen/Qwen2.5-7B-Instruct`; temperature 0; `max_tokens` 512; `max_model_len` 8192.
- Context: top **5** passages of the run, each expanded to `chunk_idx ± 1`; overlapping or adjacent windows of one document merged; neighbours joined with `chunk.drop_carry`.
- Sources labelled `[1]`…`[n]` (n ≤ 5) in retrieval rank order; a merged block takes its best rank.
- Abstention line, exactly: `I'm sorry, but I can't find that information in our database.`
- Citation parsing accepts `[2]`, `[1][3]` and `[1, 3]`; numbers outside 1..n go to `invalid_citations`. The answer text is never edited.
- `meta.json` records: model, model revision, vLLM version, sampling params, prompt SHA-256 and version, git commit, retrieval run folder name, k, neighbours, date.
- Output folder `runs/gen_r1_dev_<date>/` with `answers.jsonl` + `meta.json`; **never overwrite** an existing folder.
- Prompt tuning on dev only. No test-split questions anywhere in this plan.
- No GPU, vLLM or Drive needed by any test (CI runs `pytest -q` on ubuntu with core + dev deps).

**Plan-level calls (refining the spec, flagged for review):**
- `latency_s` per answer is dropped: vLLM batches all 100 prompts, so per-question latency isn't measurable here. `meta.json` records total `wall_s` instead. Per-stage latency is #25's job.
- `abstained` = the normalised abstention line appears in the answer **and** the answer has no valid citation. A partial answer that cites something and then says what's missing is not an abstention.
- A question with zero hits in the run skips the model and gets the abstention line directly (same as the demo's canned path), with `prompt_tokens = completion_tokens = 0`.

## Review Focus

1. **Two top-5 hits from one document that are adjacent or overlap.** Expected: one block, no passage repeated, block lists both hit IDs. Test: `test_adjacent_hits_merge_into_one_block` (Task 1).
2. **Hit at chunk 0 or the last chunk, or a one-chunk document.** Expected: window clipped, no KeyError. Test: `test_window_clipped_at_document_edges` (Task 1).
3. **Model cites a number that doesn't exist (`[7]` with 3 sources) or uses `[1, 3]` / `[1][3]`.** Expected: valid ones mapped, invalid kept separately. Test: `test_parse_citation_forms_and_invalid` (Task 3).
4. **Abstention written with a curly apostrophe, different case, or extra whitespace.** Expected: still detected. Test: `test_abstention_normalised` (Task 3).
5. **Question missing from the run, or re-running on the same day.** Expected: abstention without a model call; second save raises `FileExistsError` and the first folder is untouched. Tests: `test_question_without_hits_abstains_without_model_call`, `test_save_refuses_to_overwrite` (Task 4).

---

### Task 1: Context: passage loading and source blocks

**Files:**
- Create: `src/lumbr/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Produces:
  - `NEIGHBOURS = 1`, `K = 5`
  - `load_passages(doc_ids: set[str], passages_path: Path | str) -> dict[str, dict]`: `{passage_id: passage}` for every passage of those documents. Single streaming pass; stops early once every requested document has `n_chunks` passages.
  - `build_sources(hit_ids: list[str], by_id: dict[str, dict], neighbours: int = NEIGHBOURS) -> list[dict]`: each source is `{"n": int (1-based), "doc_id": str, "passage_ids": list[str] (the hits in this block, rank order), "window_ids": list[str] (every passage in the block, document order), "header": str, "body": str}`.

- [ ] **Step 1: Write failing tests** in `tests/test_generate.py`. Add a helper `_doc(doc_id, n_chunks)` that returns schema-shaped passage dicts with bodies `"<doc_id> chunk <i>"` and header `"[slack] #eng · T · A · 2026-01-0<i>"` (the header varies so tests can see which one is used). Tests:
  - `test_load_passages_keeps_only_requested_docs`: a JSONL with docs `a` (3 chunks), `b` (2), `c` (1); `load_passages({"a","c"}, path)` returns exactly the 4 passage IDs of `a` and `c`.
  - `test_single_hit_gets_neighbours`: hit `a#001` in a 5-chunk doc → one source, `window_ids == ["a#000","a#001","a#002"]`, `passage_ids == ["a#001"]`, `n == 1`.
  - `test_window_clipped_at_document_edges`: hits `a#000` (5 chunks) and `b#000` (1 chunk) → windows `["a#000","a#001"]` and `["b#000"]`.
  - `test_adjacent_hits_merge_into_one_block`: hits `["a#001", "x#002", "a#003"]` (a has 6 chunks) → 2 sources; source 1 has `window_ids == a#000..a#004`, `passage_ids == ["a#001","a#003"]`; source 2 is `x`. No passage ID appears in two sources.
  - `test_blocks_in_rank_order_by_best_hit`: hits `["a#004", "b#000", "a#000"]` with a 10 chunks → `a#004` and `a#000` windows don't touch, so 3 sources with `n` 1, 2, 3 in that hit order.
  - `test_body_drops_carry_and_uses_first_header`: two consecutive passages where the second body starts with the last line of the first; `body` contains that line once; `header` equals the first window passage's header.

- [ ] **Step 2: Run** `pytest tests/test_generate.py -v`. Expected: FAIL, `ModuleNotFoundError: lumbr.generate`.

- [ ] **Step 3: Implement** `load_passages` and `build_sources` in `src/lumbr/generate.py` with a module docstring that points to the spec. Merge rule: per document, windows `[max(0, i-k), min(n_chunks-1, i+k)]` merge when `next.lo <= cur.hi + 1`. Blocks sorted by the smallest hit rank they contain. `body` = first window passage's `body`, then each next one through `chunk.drop_carry(prev["body"], cur["body"])`, joined with `"\n"`. A neighbour missing from `by_id` is skipped; a missing **hit** raises `KeyError`.

- [ ] **Step 4: Run** `pytest tests/test_generate.py -v`. Expected: all PASS.

- [ ] **Step 5: Commit** `git add src/lumbr/generate.py tests/test_generate.py && git commit -m "generate: passage loading and small-to-big source blocks (#14)"`

### Task 2: Prompt file and chat messages

**Files:**
- Create: `src/lumbr/prompts/answer.json`
- Modify: `src/lumbr/generate.py`, `pyproject.toml` (package data + `generate` extra)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: source dicts from `build_sources` (Task 1).
- Produces:
  - `PROMPT_PATH: Path` (the packaged `answer.json`)
  - `load_prompt(path: Path | str = PROMPT_PATH) -> dict`: the JSON keys `version`, `system`, `user_template`, `source_template`, `abstain`, plus `sha256` (hex SHA-256 of the file's bytes).
  - `build_messages(question: str, sources: list[dict], prompt: dict) -> list[dict]`: `[{"role": "system", ...}, {"role": "user", ...}]`.

- [ ] **Step 1: Write failing tests**:
  - `test_prompt_file_is_self_consistent`: `load_prompt()` has all five keys, `version == 1`, `len(sha256) == 64`, and `prompt["abstain"] in prompt["system"]`.
  - `test_build_messages_labels_sources_in_order`: two sources → user content contains `"[1] <header1>\n<body1>"` before `"[2] <header2>\n<body2>"`, and ends with the question; system content equals `prompt["system"]`.
  - `test_prompt_hash_changes_with_content`: copy the file to `tmp_path`, change one character, and the `sha256` differs.

- [ ] **Step 2: Run** `pytest tests/test_generate.py -v`. Expected: the new tests FAIL.

- [ ] **Step 3: Create** `src/lumbr/prompts/answer.json` with this draft (Task 5 tunes it):

```json
{
  "version": 1,
  "system": "You are Lumbr, an assistant for employees of Redwood Inference. You answer questions using only the numbered sources you are given: excerpts from the company's Slack, email, tickets, documents and meeting notes.\n\nRules:\n- Use only facts stated in the sources. Never use outside knowledge and never guess.\n- After every sentence that states a fact, cite its source number in square brackets, like [2]. If it comes from two sources, cite both, like [1][3].\n- If the sources do not contain the answer, reply with exactly this sentence and nothing else: I'm sorry, but I can't find that information in our database.\n- If the sources answer only part of the question, answer that part with citations, then say which part you could not find.\n- If sources disagree, say so and cite each one. When dates show which is newer, say which is current.\n- Be concise: a few sentences or a short list, written for a coworker. Don't refer to the sources in words (\"according to source 2\"); just cite them.",
  "user_template": "Sources:\n\n{sources}\n\nQuestion: {question}",
  "source_template": "[{n}] {header}\n{body}",
  "abstain": "I'm sorry, but I can't find that information in our database."
}
```

Sources in `{sources}` are joined with `"\n\n"`.

- [ ] **Step 4: Implement** `PROMPT_PATH`, `load_prompt`, `build_messages`. In `pyproject.toml` add `[tool.setuptools.package-data]` with `lumbr = ["prompts/*.json"]`, and add the extra `generate = ["vllm<0.34"]` with the comment `# 0.34.0 fails at start-up ("No module named 'cutlass'")`.

- [ ] **Step 5: Run** `pytest tests/test_generate.py -v`. Expected: all PASS.

- [ ] **Step 6: Commit** `git add src/lumbr/prompts/answer.json src/lumbr/generate.py pyproject.toml tests/test_generate.py && git commit -m "generate: answer prompt file and chat messages (#14)"`

### Task 3: Citation parsing and abstention

**Files:**
- Modify: `src/lumbr/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: source dicts (Task 1), prompt dict (Task 2).
- Produces: `parse_answer(text: str, sources: list[dict], prompt: dict) -> dict` returning `{"citations": [{"n": int, "passage_ids": list[str], "doc_id": str}] (unique, in order of first appearance), "invalid_citations": list[int] (unique, order of appearance), "abstained": bool}`.

- [ ] **Step 1: Write failing tests**:
  - `test_parse_citation_forms_and_invalid`: 3 sources; text `"A is true [1]. B too [1, 3]. C [3][2]. D [7]."` → citation `n` order `[1, 3, 2]`, each with the source's `passage_ids` and `doc_id`; `invalid_citations == [7]`; `abstained is False`.
  - `test_abstention_exact`: text equal to `prompt["abstain"]` → `abstained is True`, `citations == []`.
  - `test_abstention_normalised`: `"  i’m SORRY, but I can’t find  that information in our database. "` → `abstained is True`.
  - `test_partial_answer_is_not_abstention`: `"The launch is in May [2]. I'm sorry, but I can't find that information in our database."` → `abstained is False`, one citation.
  - `test_non_citation_brackets_ignored`: `"Use the [beta] flag [1]."` → only `n == 1`.

- [ ] **Step 2: Run** `pytest tests/test_generate.py -v`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement** `parse_answer`. Match brackets with `r"\[(\d+(?:\s*,\s*\d+)*)\]"`. Normalise for abstention by lowercasing, turning `’ ‘` into `'`, and collapsing whitespace. `abstained` = the normalised `abstain` is in the normalised text **and** there's no valid citation.

- [ ] **Step 4: Run** `pytest tests/test_generate.py -v`. Expected: all PASS.

- [ ] **Step 5: Commit** `git add -A src/lumbr/generate.py tests/test_generate.py && git commit -m "generate: citation parsing and abstention detection (#14)"`

### Task 4: Generate loop, vLLM adapter, saving

**Files:**
- Modify: `src/lumbr/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `build_sources`, `build_messages`, `parse_answer`, `load_prompt`.
- Produces:
  - `ChatFn = Callable[[list[list[dict]]], list[dict]]`: takes a batch of message lists and returns one `{"text": str, "prompt_tokens": int, "completion_tokens": int}` per input, in order.
  - `make_vllm_chat(llm, sampling_params) -> ChatFn`: wraps `llm.chat(batch, sampling_params, use_tqdm=True)`. The text is `out.outputs[0].text`, and the token counts are `len(out.prompt_token_ids)` and `len(out.outputs[0].token_ids)`. Not unit-tested; it needs vLLM.
  - `generate(chat: ChatFn, questions: list[tuple[str, str]], run: dict[str, list[tuple[str, float]]], by_id: dict[str, dict], prompt: dict, k: int = K, neighbours: int = NEIGHBOURS) -> tuple[list[dict], float]`: returns (records, wall_s). Each record is `{"question_id", "question", "answer", "abstained", "citations", "invalid_citations", "sources": [{"n", "doc_id", "passage_ids", "window_ids"}], "prompt_tokens", "completion_tokens"}`, in input question order. **One** `chat` call for all questions that have hits.
  - `needed_doc_ids(run, question_ids: Iterable[str], k: int = K) -> set[str]`: the doc IDs of the top-k hits of those questions (feeds `load_passages`).
  - `save_answers(records: list[dict], runs_dir: Path, meta: dict, *, tag: str = "gen_r1", split: str = "dev", date: str | None = None) -> Path`: writes `{tag}_{split}_{date}/answers.jsonl` + `meta.json`. `meta.json` = caller's `meta` plus `git_commit`, `date`, `n_questions`, `n_abstained`, `n_invalid_citations`. Raises `FileExistsError` if the folder exists.

- [ ] **Step 1: Write failing tests**, using a fake chat that records its calls and returns scripted texts:
  - `test_generate_end_to_end_with_fake_chat`: 2 questions, each with hits in a small passage set. The fake answers `"X [1]."` and the abstain line. Records come back in question order. Record 1 has `citations[0]["doc_id"]` equal to the hit's doc and `abstained False`. Record 2 has `abstained True`. Exactly one chat call, with 2 message lists, and each user message contains its question.
  - `test_only_top_k_hits_used`: a run with 8 hits and `k=5` → the sources' `passage_ids` together hold only the first 5 hit IDs.
  - `test_question_without_hits_abstains_without_model_call`: the question isn't in `run` → `answer == prompt["abstain"]`, `abstained True`, `sources == []`, tokens 0, and the fake was called with no message list for it (or not called at all when it's the only question).
  - `test_needed_doc_ids`: returns only the doc IDs from the top-k hits of the requested questions.
  - `test_save_writes_answers_and_meta`: `save_answers(records, tmp_path, {"model": "m"}, date="2026-11-23")` → folder `gen_r1_dev_2026-11-23`. `answers.jsonl` has one line per record and round-trips. `meta.json` has `model == "m"`, `n_questions`, `n_abstained`, `git_commit` and `date`.
  - `test_save_refuses_to_overwrite`: a second call with the same date raises `FileExistsError`, and the first `answers.jsonl` is unchanged.

- [ ] **Step 2: Run** `pytest tests/test_generate.py -v`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement** `ChatFn`, `make_vllm_chat`, `needed_doc_ids`, `generate` and `save_answers`. Get the git commit the way `index_bm25.save_run` does (`git rev-parse --short HEAD`, falling back to `"unknown"`). Time `wall_s` with `time.perf_counter()` around the chat call. Import vLLM nowhere at module level.

- [ ] **Step 4: Run** `pytest -q` (the whole suite). Expected: all PASS, nothing skipped that passed before.

- [ ] **Step 5: Commit** `git add src/lumbr/generate.py tests/test_generate.py && git commit -m "generate: batch generate loop, vLLM adapter, run saving (#14)"`

### Task 5: Notebook 17, dev run and prompt tuning

**Files:**
- Create: `notebooks/17_john_generator.ipynb` (copy `notebooks/00_template.ipynb`, keeping its bootstrap cells unchanged)
- Modify: `README.md` (Layout: add `generate.py`, `prompts/answer.json`, notebook 17)

**Interfaces:**
- Consumes: everything from Tasks 1–4. Also `paths.latest_run("r1_bm25", "dev")`, `index_bm25.read_run`, `data.load_questions()`, `splits/dev_question_ids.txt`, and `paths.PROCESSED / "passages.jsonl"`.

- [ ] **Step 1: Write the notebook cells**, in order:
  1. A header markdown cell: owner John, issue #14, GPU L4, the output folder, and a results table (round, changes, hand-judged good/ok/bad) like notebook 10's.
  2. The two bootstrap cells from the template.
  3. `!pip install -q -e ".[generate]"` and print the vLLM version.
  4. Config: `MODEL = "Qwen/Qwen2.5-7B-Instruct"`. Resolve `MODEL_REVISION` with `huggingface_hub.model_info(MODEL).sha` and pass it as `revision=` to `LLM` so the weights are pinned. Also set `R1_DIR = paths.latest_run("r1_bm25", "dev")`, and `SAMPLING = dict(temperature=0.0, max_tokens=512)`.
  5. Load the dev questions as `(question_id, question)` pairs in `dev_question_ids.txt` order, the R1 run, `prompt = generate.load_prompt()`, and `by_id = generate.load_passages(generate.needed_doc_ids(run, qids), passages_path)`.
  6. `llm = LLM(model=MODEL, revision=MODEL_REVISION, max_model_len=8192, gpu_memory_utilization=0.90, seed=0)`, then `chat = generate.make_vllm_chat(llm, SamplingParams(**SAMPLING))`.
  7. `records, wall_s = generate.generate(chat, questions, run, by_id, prompt)`.
  8. Inspection: abstention count overall and on `question_type == "info_not_found"`, count of invalid citations, the prompt token distribution (max must be < 8192 − 512), and 20 answers printed with question, answer and source headers. Include every dev `info_not_found` question.
  9. Save: `generate.save_answers(records, paths.RUNS, meta)`. `meta` holds `model`, `model_revision`, `vllm_version`, `sampling`, `prompt_sha256`, `prompt_version`, `retrieval_run` (`R1_DIR.name`), `k`, `neighbours`, `wall_s` and `split`.

- [ ] **Step 2: Verify locally without a GPU.** Run `python -c "import json; json.load(open('notebooks/17_john_generator.ipynb', encoding='utf-8'))"`, then `pytest -q`. Expected: the notebook parses and all tests pass.

- [ ] **Step 3: Commit and push** `git add notebooks/17_john_generator.ipynb README.md && git commit -m "Notebook 17: answer generator dev run (#14)" && git push -u origin john/14-answer-generator`

- [ ] **Step 4 (John, on Colab L4): run notebook 17 top to bottom.** Expected: 100 records, a `gen_r1_dev_<date>/` folder on Drive, and no prompt over the token limit.

- [ ] **Step 5 (John + Claude): tune the prompt on dev.** Judge the 20 printed answers by hand: are claims supported by the cited source, does it abstain correctly, is it concise? If a round changes the prompt, bump `version` in `answer.json`, add a row to the notebook's table, rerun, and commit. Stop when the answers look right. Then the frozen prompt's SHA-256 is the one #21/#22 record.

- [ ] **Step 6: Open the PR** against `main`. Closes #14, and links the spec, the plan and the Drive run folder name.
