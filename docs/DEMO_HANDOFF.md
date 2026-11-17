# "First Day at Redwood" demo: handoff

**As of 2026-11-17.** Owner: John. This note says where the demo stands and what's left. The full design and
plans are linked at the bottom.

## Where things live

| What | Where |
|---|---|
| Demo site code (UI, game, API routes) | `personal_site` repo, branch **`john/lumbr-demo`** (pushed, 14 commits, not merged) |
| Next.js security upgrade (16.4.3 → 16.4.6) | `personal_site` repo, branch **`john/next-security-upgrade`** (pushed, not merged). **Merge this first.** It fixes the whole site. |
| Design spec | `docs/superpowers/specs/2026-11-16-first-day-demo-design.md` (this repo) |
| Site plan (Tasks 1–10) | `docs/superpowers/plans/2026-11-16-first-day-demo-site.md` |
| Backend + content plan (Tasks B1–B6) | `docs/superpowers/plans/2026-11-16-first-day-demo-backend.md` |
| Demo URL once merged | `/projects/lumbr-enterprise-rag/demo` (the project page has a "Try your first day" button) |

## Status

**Done (site plan Tasks 1–9).** A full round plays end to end on **fake data**:
- Retro desktop with a drawn wallpaper, a "redwood" start button, and draggable, scrolling windows.
- Colleagues start arriving 3 s after "Let's go"; at most 2 wait at once; they nudge if you're slow and show "typing…" for 1.5–3 s before reacting.
- The Lumbr icon (a log with a leaf) wiggles while someone is waiting. Lumbr answers with clickable citations and follow-ups.
- Replies are judged correct, partly right or wrong. The round ends in a letter grade with a Performance Review or a **PIP**, plus the full answer key and sources.
- Phone layout (< 768 px): chats only, with Lumbr as a contact.
- Safety: off-topic questions get the canned "can't find" line without calling the LLM (the professor's similarity filter), plus a per-IP rate limit, a daily spend ceiling ($0.33/day ≈ $10/month), and one ~8 s deadline per question with a saved-answer fallback. Production never falls back to fakes.
- 82 unit tests (`npm test` in `personal_site`), lint and types are clean, and a fresh whole-branch review was done with its 3 Important findings fixed.

**Not started:**
- **Site Task 10:** real questions, cached answers, going live.
- **The whole backend plan (B1–B6):** retrieval service on Vercel, ONNX encoder, int8 index, content bundle, threshold tuning.

## How to try it

- **Locally:** in `personal_site`, check out `john/lumbr-demo` and run `npm run dev`, then open `/projects/lumbr-enterprise-rag/demo`. With no env vars set, everything runs on fakes.
- **Vercel preview:** each push to the branch builds a preview (they're behind Vercel login). Previews run in production mode, so they need **`LUMBR_DEMO_FAKES=1` set for Preview only** in Vercel (Settings → Environment Variables), or Lumbr shows "coffee break." The production deployment ignores this flag.
- **Known quirk:** next@16.4.3 halts every other `next dev` or `next build` run with a security notice. Retrying continues. It goes away once the upgrade branch is merged and `john/lumbr-demo` is rebased on it.

## What's blocking what

| Next step | Blocked on | Who |
|---|---|---|
| B1–B3: index, API, ONNX encoder, first deploy on the v0 sample | nothing (Drive is reachable) | John |
| B3 step 1: match the study's R2 query encoding (prefix, pooling) | Patricia's retrieval code | Patricia → John |
| B4: real `questions.json` | **dev split file** `processed/dev_question_ids.txt` | Patricia |
| B4: colleague DM rewrites | John reviewing `demo/content/colleague_dms.csv` by hand | John |
| B5: choose the retriever, tune the threshold, run the full index | the study's results (~Nov 23) | team |
| B6: hybrid + reranker in the demo | the team's finished cross-encoder reranker (the demo only *serves* it; building it is study work) | team |
| Site Task 10: cached answers, About panel, Upstash, LLM keys, provider hard cap | B4 + B5 | John |

## Open decisions

- **Does response time affect the grade?** John is deciding with Patricia and Jieun. It's one switch: `TIME_COUNTS_IN_GRADE` in `src/lib/lumbr-demo/config.ts` (currently `false`). Response times are already recorded.
- **Copy to write (placeholders for now):** easter eggs in `src/data/lumbr-demo/easter-eggs.ts` and colleague reactions in `src/data/lumbr-demo/reactions.ts`.
- **Lumbr answers instantly on fake data.** Optionally give it a short "thinking" delay like the colleagues have.

## Going live checklist (site Task 10)

1. Replace `src/data/lumbr-demo/questions.json` with the real dev-split bundle from B4. Never use test questions.
2. In Vercel (Production and Preview): `LUMBR_BACKEND_URL`, `LUMBR_BACKEND_KEY`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_PRICE_IN_PER_M`, `LLM_PRICE_OUT_PER_M` (must be real positive numbers, or the LLM stays off), `LUMBR_DAILY_BUDGET_USD`. Provision Upstash Redis from the Vercel Marketplace (`UPSTASH_REDIS_REST_URL`/`_TOKEN`). See `.env.example`.
3. **Set a hard spend cap at the LLM provider too.** The in-app ceiling can overshoot by a few cents under bursts (accepted trade-off).
4. Run `npx tsx scripts/precompute-lumbr-answers.ts` to fill the cached answers.
5. Fill `src/data/lumbr-demo/about.ts` with the served retriever, its study score, and how it differs (int8 index, rerank depth, multi-turn follow-ups not measured by the study).
6. Play a round on the preview (desktop and phone), then merge.

## Known minor issues (deferred, from review)

- An in-flight Lumbr answer can land in a new round after "Start over".
- The Lumbr double-submit guard reads a stale closure, so very fast repeat submits can both go through.
- The notice auto-hide timer isn't cleared on restart (harmless).
- A client can put up to about 3.6K characters of made-up history into its own prompt. It only affects that visitor's session.
- The Redis rate limiter could lock out an IP if `pexpire` fails after `incr`. Fix with a pipeline or `SET NX PX`.
- The same colleague name can appear on several questions, giving repeated contact rows.
