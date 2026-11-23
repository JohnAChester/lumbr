# "First Day at Redwood" demo: handoff

**As of 2026-11-23 (#18).** Owner: John. This note says where the demo stands and what's left. The full design and
plans are linked at the bottom.

## Where things live

| What | Where |
|---|---|
| Demo site code (UI, game, API routes) | `personal_site` repo, branch **`john/lumbr-demo`** (pushed, not merged) |
| Next.js security upgrade (16.4.3 → 16.4.6) | `personal_site` repo, branch **`john/next-security-upgrade`** (pushed, not merged). **Merge this first.** It fixes the whole site. |
| Search service (R2 retrieval, similarity filter, the study's sources + messages) | this repo, `demo/backend/` (README there) |
| Build scripts (artifacts, threshold, questions, parse cases) | this repo, `demo/build/`; content in `demo/content/` |
| Demo design | `docs/superpowers/specs/2026-11-16-first-day-demo-design.md` |
| #18 design + plan (real retrieval and generator) | `docs/superpowers/specs/2026-11-23-interface-skeleton-design.md`, `docs/superpowers/plans/2026-11-23-interface-skeleton.md` |
| Older plans | `docs/superpowers/plans/2026-11-16-first-day-demo-site.md` (site Tasks 1–10), `...-backend.md` (B1–B6; the #18 plan replaces B1–B4) |
| Demo URL once merged | `/projects/lumbr-enterprise-rag/demo` (the project page has a "Try your first day" button) |

## Status

**Done (#18): a dev question works end to end locally** with real retrieval and the study's generator.
- Search service over the **1K-doc v0 sample** (5,233 passages): R2 (`bge-base-en-v1.5`) through an ONNX encoder that
  matches sentence-transformers (cosine ≥ 0.999, same top 5 on 20 dev questions), ~18 ms per `/retrieve` locally.
- `/retrieve` returns the top-5 hits, their **small-to-big sources** and the **finished chat messages**, built by
  `lumbr.generate` itself, so the demo's context and prompt are the study's (prompt v4).
- The site sends those messages to the LLM (temperature 0, 512 tokens) and parses answers with a TypeScript port of
  `parse_answer`, pinned by `parse_cases.json` exported from the Python tests. `[1, 3]` shows as two chips, out-of-range
  numbers are dropped, and abstentions with up to 2 inserted words count as the canned line.
- **Generator: Qwen3.5-9B** with thinking off (`qwen/qwen3.5-9b` on OpenRouter, bf16 providers only). Qwen2.5-7B-Instruct
  was retired by the hosts; the study switches too (branch `john/14b-qwen35`, dev validation pending on Colab).
- **15 colleague questions** from the dev split: the 11 whose gold docs are all in the sample, plus the 4 `info_not_found`.
  Colleague DMs are drafts in `demo/content/colleague_dms.csv` (John to review).
- **Provisional threshold 0.552** (lowest top score of the 11 answerable dev questions, minus 0.02). All 4
  `info_not_found` questions pass it (0.67–0.77), so the generator has to abstain on them, which it did in the check.
- End-to-end check (2026-11-23): qst_0019, qst_0114, qst_0466 answered correctly with the gold doc cited (0.4–1.1 s);
  qst_0481 returned the canned line; a follow-up was answered from the earlier context.

Earlier (site Tasks 1–9): retro desktop, colleagues arriving and nudging, the wiggling Lumbr icon, reply grading, letter
grade with Performance Review or **PIP**, phone layout, rate limit, daily spend ceiling ($0.33/day), one ~8 s deadline
per question with a saved-answer fallback. Production never falls back to fakes.

**Not done:**
- **Deploy the search service** (Vercel project rooted at `demo/backend`, artifacts from a private HF repo).
- **Full index and retriever choice (B5)** after #22; hybrid + reranker serving (B6).
- **Site Task 10:** cached answers, About panel, Upstash, going live.

## How to try it

- **Locally with real data:** build the artifacts once and run the service (`demo/backend/README.md`), then in
  `personal_site` on `john/lumbr-demo` put the LLM variables in `.env.local` and run
  `LUMBR_BACKEND_URL=http://127.0.0.1:8787 LUMBR_BACKEND_KEY=dev npm run dev`. Use `127.0.0.1`, not `localhost`
  (Windows tries IPv6 first and adds ~2 s per call).
- **Locally on fakes:** with no backend or LLM variables, everything runs on fakes (fake passages, a stand-in LLM).
  The round now uses the real questions, so fake mode mostly abstains; tests use `tests/lumbr-demo/fixtures/fake-questions.json`.
- **Vercel preview:** previews run in production mode, so they need **`LUMBR_DEMO_FAKES=1` set for Preview only**, or
  Lumbr shows "coffee break". Don't add the LLM variables to Preview until the service is deployed: with them set, a
  preview makes paid calls against fake passages.
- **Known quirks:** next@16.4.3 halts every other `next dev` or `next build` run with a security notice (retry). Only one
  `next dev` can run per project: stop an old one before starting another.

## What's blocking what

| Next step | Blocked on | Who |
|---|---|---|
| Review the 15 colleague DMs | nothing | John |
| Qwen3.5-9B dev validation (load test on L4, v4 over 100 dev, compare to Qwen2.5 v4, hand-check 20) | Colab session | John |
| Deploy the search service + set Preview/Production env vars | John's Vercel + HF accounts | John |
| B5: choose the served retriever, full index, real threshold (with an off-topic set) | #22 test results | team |
| B6: hybrid + reranker in the demo | B5 | team |
| Site Task 10: cached answers, About panel, Upstash, provider hard cap | deploy + B5 | John |

## Open decisions

- **Does response time affect the grade?** John is deciding with Patricia and Jieun. It's one switch: `TIME_COUNTS_IN_GRADE` in `src/lib/lumbr-demo/config.ts` (currently `false`). Response times are already recorded.
- **Copy to write (placeholders for now):** easter eggs in `src/data/lumbr-demo/easter-eggs.ts` and colleague reactions in `src/data/lumbr-demo/reactions.ts`.

## Going live checklist (site Task 10)

1. ~~Replace `questions.json` with the real dev-split bundle.~~ Done for the v0 sample (15 questions); rebuild with
   `demo/build/build_questions.py` once the full index is served. Never use test questions.
2. In Vercel (Production and Preview): `LUMBR_BACKEND_URL`, `LUMBR_BACKEND_KEY`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (`qwen/qwen3.5-9b`), `LLM_PRICE_IN_PER_M` (0.17), `LLM_PRICE_OUT_PER_M` (0.25), `LUMBR_DAILY_BUDGET_USD`. Provision Upstash Redis from the Vercel Marketplace (`UPSTASH_REDIS_REST_URL`/`_TOKEN`). See `.env.example`.
3. **Set a hard spend cap at the LLM provider too** (an OpenRouter key credit limit). The in-app ceiling can overshoot by a few cents under bursts (accepted trade-off).
4. Run `npx tsx scripts/precompute-lumbr-answers.ts` to fill the cached answers.
5. Fill `src/data/lumbr-demo/about.ts` with the served retriever, its study score, the generator (Qwen3.5-9B, hosted bf16), and how it differs (index format, rerank depth, multi-turn follow-ups not measured by the study).
6. Play a round on the preview (desktop and phone), then merge.

## Generator prompt (from #14)

The demo never holds a copy of the prompt. The search service builds the chat messages with
`lumbr.generate.build_messages` from `src/lumbr/prompts/answer.json`, and `/health` and `/retrieve` report its
`prompt_sha256`. Check that hash against the `meta.json` of the study run the demo serves. Follow-ups keep the
study's system prompt and pass earlier turns as chat history, with their `[n]` markers stripped. Citation parsing
on the site is pinned to the study's by `tests/lumbr-demo/parse_cases.json`; regenerate it with
`demo/build/export_parse_cases.py` whenever `parse_answer` changes.

## Known minor issues (deferred, from review)

- An in-flight Lumbr answer can land in a new round after "Start over".
- The Lumbr double-submit guard reads a stale closure, so very fast repeat submits can both go through.
- The notice auto-hide timer isn't cleared on restart (harmless).
- A client can put up to about 3.6K characters of made-up history into its own prompt. It only affects that visitor's session.
- The Redis rate limiter could lock out an IP if `pexpire` fails after `incr`. Fix with a pipeline or `SET NX PX`.
- The same colleague name can appear on several questions, giving repeated contact rows.
