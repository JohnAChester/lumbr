# Answer generator (#14): design

Owner: John. Status: draft for approval. Depends on: R1 dev run (#8, done).

## Intent

The generator writes Lumbr's answer: the text a user sees. It is held **fixed** across all six retrieval
setups, so differences in answer quality come from retrieval alone (proposal Sec. 3). The same prompt and
model also power the demo through a pay-per-token API. So the priority is **certainty and reproducibility**:
one implementation, one prompt, every setting recorded.

## Goals

- Given a question and a retrieval run, answer from the **top 5 passages** only, cite each claim, or abstain.
- Shared, tested code in `src/lumbr/` that #21/#22 (all setups) and the demo reuse unchanged.
- Done when: all 100 dev questions are answered over the R1 (BM25) dev run and saved as a run folder, and a hand
  read of about 20 answers (including the dev "Info Not Found" questions) looks right.

## Non-goals

- Scoring answers (LLM judge, faithfulness, abstention metrics): #23/#24.
- Running the other retrievers or the test split: #21/#22.
- Demo wiring and conversation follow-ups: the demo spec.

## Anti-goals

- Two copies of the prompt that drift apart (study vs. demo, or setup vs. setup).
- An answer that looks uncited because the model mangled a long ID.
- Ever tuning the prompt on test questions (dev only, per the README's data rule).

## Approaches

- **A. Shared module + notebook (chosen).** Pure-Python logic in `src/lumbr/generate.py` with pytest tests;
  the prompt in its own file; a GPU notebook that runs vLLM. More setup now, but every later run and the demo
  call the same code. Picked by John for certainty and reproducibility.
- **B. All in the notebook.** Fastest first answer, but the logic would be copied into #21/#22 and the demo,
  which breaks "generator held fixed".
- *Dropped:* vLLM as an OpenAI-style server on Colab. More moving parts; the shared prompt file already gives
  the demo the same prompt.

## Decisions

Decided with John:

1. **Context = top 5 passages + neighbours** (small-to-big, `docs/SCHEMA.md` D16), not whole documents. Each
   hit is expanded to `chunk_idx ± 1` in its document; adjacent or overlapping windows from the same document
   are merged so no passage appears twice. Neighbours are joined with `chunk.drop_carry`.
2. **Numbered citations.** The model sees sources labelled `[1]`…`[n]` (n ≤ 5) and cites those numbers. Code
   maps each number back to its hit passage ID(s) and doc ID. This replaces the issue's literal `[doc-ID]`
   format, because 7–8B models garble 36-character hashes.
3. **Approach A** (above).

Claude's calls (correct any of these):

4. **Model: `Qwen/Qwen2.5-7B-Instruct`**, vLLM offline `LLM.chat`, L4 GPU. Ungated (Llama needs a licence
   approval), Jieun's notebook already runs Qwen on vLLM, and hosted versions exist for the demo.
   Temperature 0, `max_tokens` 512, `max_model_len` 8192 (5 windows of up to 3 × 512 tokens fit).
5. **Pinned for reproducibility:** model revision (commit hash), vLLM version, sampling params, prompt SHA-256,
   git commit and the retrieval run's folder, all in `meta.json`.
6. **Source block format.** Each block starts with its label and the hit's one-line header (source, channel,
   title, author, date), then the merged bodies. Headers matter for "who" and "when" questions.
   Blocks are in retrieval rank order. When two hits merge into one window, the block takes the better rank.
7. **Abstention = one exact sentence**, the demo's canned line: *"I'm sorry, but I can't find that information
   in our database."* Detected by normalised string match, so the demo and the metrics agree on what an
   abstention is. If the answer is partial, the model answers what it can, cites it, and says what is missing.
8. **Citation parsing** accepts `[2]` and `[1, 3]`. Numbers outside 1..n are kept as `invalid_citations`
   (a signal for #23). No other post-editing of the answer text.
9. **Prompt file:** `src/lumbr/prompts/answer.json`, holding the system prompt, the source-block template, and the
   abstention line. The demo copies it and checks the SHA-256 recorded in `meta.json`.
10. **Passage lookup:** stream `passages.jsonl` once and keep only the documents the run's top-5 hits touch
    (about 500 docs for 100 questions), instead of loading 772K passages into memory.
11. **Output:** `runs/gen_r1_dev_<date>/answers.jsonl` + `meta.json`, never overwriting (same convention as
    `index_bm25.save_run`). One line per question: `question_id`, `question`, `answer`, `abstained`,
    `citations` (`[{n, passage_ids, doc_id}]`), `invalid_citations`, `sources` (what the model saw, with
    passage IDs), `prompt_tokens`, `completion_tokens`, `latency_s`.
12. **Prompt iteration** on dev only, a few rounds judged by hand, with a short table in the notebook like
    Jieun's #9 notebook. After sign-off the prompt is frozen and its hash is what #21/#22 record.

## Pieces

| piece | where | GPU |
|---|---|---|
| expand + merge windows, build source blocks | `generate.py` | no |
| build chat messages from the prompt file | `generate.py` | no |
| parse citations, detect abstention, map to IDs | `generate.py` | no |
| load needed passages from the JSONL | `generate.py` | no |
| `generate(llm, questions, run, k=5)` and `save_answers(...)` | `generate.py` (takes an injected LLM, so tests use a fake) | no |
| tests | `tests/test_generate.py`, on `tests/fixtures/` | no |
| model load, dev run, hand read | `notebooks/17_john_generator.ipynb` | L4 |

## Left to the builder

The exact prompt wording (it starts as a draft and gets tuned in the notebook), function signatures, and the
exact notebook number.
