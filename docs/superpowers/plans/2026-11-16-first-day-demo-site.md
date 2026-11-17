# First Day at Redwood — Site (personal_site) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A playable "First Day at Redwood" round on John's site — retro desktop (phone: chat list), colleagues, Lumbr chat with citations, LLM-judged replies, letter grade and Review/PIP ending — working end to end against fakes, switchable to the real backend and LLM by env vars.

**Architecture:** Pure TypeScript game logic (reducer, round picking, grading, citations) in `src/lib/lumbr-demo/`, unit-tested with Vitest. Server logic (limits, spend, retrieval client, LLM calls, ask pipeline) in `src/lib/lumbr-demo/server/` behind injectable deps, exposed by Node route handlers under `src/app/api/lumbr/`. React client components in `src/components/lumbr-demo/` render a full-screen page at `/projects/lumbr-enterprise-rag/demo`.

**Tech Stack:** Next.js 16.4 (App Router, Node runtime), React 19, Tailwind 4, TypeScript, Vitest (new dev dep), Upstash Redis via Vercel Marketplace (spend + rate limit), any OpenAI-compatible pay-per-token chat API.

**Spec:** `lumbr/docs/superpowers/specs/2026-11-16-first-day-demo-design.md`. **Companion plan:** `lumbr/docs/superpowers/plans/2026-11-16-first-day-demo-backend.md` (owns the backend and `questions.json`).

## Global Constraints

- Repo: `/Users/johnchester/Developer/personal_site`. Before writing Next.js code, read the relevant guide in `node_modules/next/dist/docs/` (AGENTS.md rule; this Next.js has breaking changes).
- Node runtime only; never `export const runtime = 'edge'`.
- Route: `/projects/lumbr-enterprise-rag/demo`, its own full-screen page (covers the site nav/footer).
- Welcome text, verbatim: "Welcome to your first day at Redwood Inference! We're so excited to have you on the team. We're fast-paced, so all our new employees must hit the ground running. If you have any questions, Lumbr is here to help. We expect great things from you!"
- First colleague arrives **3 s** after the welcome.
- Canned no-match line, verbatim: "I'm sorry, but I can't find that information in our database."
- Round length config default **6** (allowed 5–10); exactly **one** Info-Not-Found question per round.
- Scoring: correct 1, partial 0.5, wrong 0, averaged over judged questions. A ≥ 90%, B ≥ 75%, C ≥ 60%, D ≥ 45%, F below. Pass = A–C ("First-Day Performance Review"); fail = D/F ("Performance Improvement Plan").
- `TIME_COUNTS_IN_GRADE = false` (open team decision; one switch). If true, the clock pauses while Lumbr is working.
- Limits: question ≤ **300** chars, ≤ **30** Lumbr questions per round, per-IP rate limit, daily spend ceiling (`LUMBR_DAILY_BUDGET_USD`, default `0.33` ≈ $10/month).
- Retro desktop is an **original homage**: no Microsoft logos, no Bliss photo; start button reads "redwood".
- Lumbr icon: fat, stylized log tilted 45° with a small leaf, inline SVG; leaf wiggles while a colleague waits.
- Phones (< 768px, Tailwind `md`): chat list only, Lumbr is a contact; banners at top; no easter eggs.
- No visitor questions/replies are persisted; no visitor identifiers stored (IP is used only as a rate-limit key with a TTL).
- Never use benchmark test questions anywhere; content comes only from `questions.json` (dev split).

## Review Focus

1. **Backend cold, slow or down** — the visitor never waits more than ~8 s: serve a cached answer if the question matches a waiting colleague's, else an in-world "Lumbr is waking up — try again in a moment." (test in Task 5).
2. **Spend store unreachable or unconfigured in production** — live mode fails *closed* (coffee-break message), never unlimited (test in Task 4).
3. **Forged context** — a client that posts passage text or huge history must not get it into the prompt; the server accepts only passage IDs and re-fetches them, and caps history (test in Task 5).
4. **Every reply ungraded** (judge down all round) — ending shows "Ungraded", not F or a PIP (test in Task 2).
5. **Double or empty replies** — a second reply to the same colleague and whitespace-only replies are ignored, both client-side (reducer) and by the judge route (tests in Tasks 3 and 6).

---

## File Structure

```
src/lib/lumbr-demo/
  types.ts            shared types (DemoQuestion, Passage, Citation, Verdict, …)
  config.ts           every constant from Global Constraints
  round.ts            pickRound()
  grade.ts            scoreRound()
  citations.ts        numberCitations()
  match.ts            matchWaitingQuestion() (safety-net similarity)
  game.ts             gameReducer() + initialState()
  server/
    limits.ts         validateQuestion(), SpendStore + RateLimiter (redis + memory impls)
    retrieval.ts      RetrievalClient (http + fake)
    llm.ts            LlmClient (OpenAI-compatible + fake), prompts
    ask.ts            askLumbr() pipeline
    judge.ts          judgeReply()
    deps.ts           buildDeps() from env
src/app/api/lumbr/{ask,judge,warm}/route.ts
src/app/projects/[slug]/demo/page.tsx
src/components/lumbr-demo/  demo-app.tsx, desktop.tsx, window-frame.tsx, taskbar.tsx, toast.tsx,
                            desktop-icons.tsx, messenger.tsx, lumbr-chat.tsx, citation-popover.tsx,
                            lumbr-log-icon.tsx, ending-document.tsx, mobile-shell.tsx, about-window.tsx
src/data/lumbr-demo/  questions.json (fixture now, real file from backend plan), easter-eggs.ts, about.ts
tests/lumbr-demo/     *.test.ts (Vitest)
```

---

### Task 1: Vitest, types, config, fixture

**Files:**
- Modify: `package.json` (devDep `vitest`, script `"test": "vitest run"`)
- Create: `vitest.config.ts` (alias `@` → `./src`, environment `node`, include `tests/**/*.test.ts`)
- Create: `src/lib/lumbr-demo/types.ts`, `src/lib/lumbr-demo/config.ts`, `src/data/lumbr-demo/questions.json`
- Test: `tests/lumbr-demo/fixture.test.ts`

**Interfaces — Produces:**
```ts
export type Category = string;                       // benchmark question_type
export type Passage = { passage_id: string; doc_id: string; source: string; title: string;
  channel: string | null; author: string | null; date: string | null;
  header: string; body: string; score: number | null };
export type Citation = { n: number; doc_id: string; passage: Passage };
export type Verdict = "correct" | "partial" | "wrong";
export type DemoQuestion = { id: string; category: Category; infoNotFound: boolean;
  colleague: string;            // display name from corpus authors
  dm: string;                   // casual colleague phrasing shown in the messenger
  question: string;             // benchmark question text
  answerFacts: string[]; goldAnswer: string;
  goldSources: { doc_id: string; header: string }[];
  cached: { text: string; citations: Citation[] } | null };   // safety net
export type LumbrAnswer = { kind: "answer" | "canned" | "cached" | "waking" | "limited";
  text: string; citations: Citation[]; contextIds: string[] };
```
`config.ts` exports: `WELCOME_TEXT`, `CANNED_LINE`, `FIRST_COLLEAGUE_DELAY_MS = 3000`, `ROUND_LENGTH = 6`, `MAX_WAITING = 2`, `NEXT_COLLEAGUE_DELAY_MS = [8000, 15000]`, `NUDGE_AFTER_MS = [45000, 90000]`, `NUDGES = ["??", "hello?? standup in 5"]`, `TIME_COUNTS_IN_GRADE = false`, `MAX_QUESTION_CHARS = 300`, `MAX_LUMBR_QUESTIONS = 30`, `MAX_CONTEXT_PASSAGES = 8`, `MAX_HISTORY_TURNS = 6`, `BACKEND_TIMEOUT_MS = 8000`, `GRADE_CUTOFFS = { A: 0.9, B: 0.75, C: 0.6, D: 0.45 }`, `MOBILE_BREAKPOINT_PX = 768`.

- [ ] **Step 1:** Install: `npm i -D vitest` and add the script and `vitest.config.ts`.
- [ ] **Step 2: Write the failing test** `tests/lumbr-demo/fixture.test.ts`: loads `questions.json`; asserts ≥ 8 entries, ≥ 2 with `infoNotFound: true`, ≥ 3 distinct categories, every non-INF entry has `answerFacts.length > 0` and `goldSources.length > 0`, every `id` unique.
- [ ] **Step 3:** Run `npm test` → FAIL (file missing).
- [ ] **Step 4:** Write `types.ts`, `config.ts` (values above, text verbatim from Global Constraints), and a hand-written **fake** `questions.json` (10 Redwood-flavoured questions, 2 INF, `cached` filled for 3) clearly marked fake via ids `fake-01…`.
- [ ] **Step 5:** `npm test` → PASS. `npm run lint` clean.
- [ ] **Step 6:** Commit `feat(lumbr-demo): vitest, types, config, fixture questions`.

---

### Task 2: Round picking and grading

**Files:** Create `src/lib/lumbr-demo/round.ts`, `src/lib/lumbr-demo/grade.ts`; Test `tests/lumbr-demo/round.test.ts`, `tests/lumbr-demo/grade.test.ts`

**Interfaces — Produces:**
- `pickRound(pool: DemoQuestion[], n: number, rng: () => number): DemoQuestion[]`
- `scoreRound(verdicts: (Verdict | "ungraded")[]): { judged: number; percent: number | null; letter: "A"|"B"|"C"|"D"|"F"|null; pass: boolean | null }`

- [ ] **Step 1: Failing tests.**
  - `pickRound` with a seeded rng: returns `n` distinct questions; exactly one `infoNotFound`; no category appears more than `ceil((n-1)/2)` times among the others when the pool allows; the INF question is never first; throws if `n` outside 5–10.
  - `scoreRound(["correct","correct","partial","wrong","correct","correct"])` → percent `0.75`, letter `"B"`, pass `true`.
  - `scoreRound(["wrong","wrong","partial","correct","wrong","wrong"])` → letter `"F"`, pass `false`.
  - boundaries: 0.9 → A, 0.45 → D (pass false), 0.6 → C (pass true).
  - `"ungraded"` entries are excluded from the average; all ungraded → `{ judged: 0, percent: null, letter: null, pass: null }` (Review Focus 4).
- [ ] **Step 2:** `npm test` → FAIL.
- [ ] **Step 3:** Implement. `pickRound`: shuffle (Fisher–Yates with `rng`), take one INF, then round-robin across categories for the rest, place INF at a random index ≥ 1.
- [ ] **Step 4:** `npm test` → PASS.
- [ ] **Step 5:** Commit `feat(lumbr-demo): round picking and letter grading`.

---

### Task 3: Game reducer

**Files:** Create `src/lib/lumbr-demo/game.ts`; Test `tests/lumbr-demo/game.test.ts`

**Interfaces — Consumes:** Task 1 types, Task 2 `scoreRound`. **Produces:**
```ts
export type ThreadMsg = { from: "colleague" | "me"; text: string; at: number };
export type GameState = { phase: "welcome" | "playing" | "ended"; round: DemoQuestion[];
  arrived: string[]; threads: Record<string, ThreadMsg[]>; nudges: Record<string, number>;
  replied: Record<string, string>; verdicts: Record<string, Verdict | "ungraded">;
  lumbr: { messages: { from: "me" | "lumbr"; answer?: LumbrAnswer; text: string }[];
           contextIds: string[]; asked: number }; startedAt: number | null };
export type GameEvent =
  | { type: "START"; at: number } | { type: "ARRIVE"; id: string; at: number }
  | { type: "NUDGE"; id: string; at: number } | { type: "REPLY"; id: string; text: string; at: number }
  | { type: "VERDICT"; id: string; verdict: Verdict | "ungraded" }
  | { type: "LUMBR_ASK"; text: string } | { type: "LUMBR_ANSWER"; answer: LumbrAnswer };
export function initialState(round: DemoQuestion[]): GameState;
export function gameReducer(s: GameState, e: GameEvent): GameState;
export function waitingIds(s: GameState): string[];      // arrived, not yet replied
export function nextArrival(s: GameState): string | null; // next unarrived id if waiting < MAX_WAITING
```

- [ ] **Step 1: Failing tests.**
  - START moves `welcome → playing`; ARRIVE adds the colleague's `dm` as the first thread message.
  - `nextArrival` returns null when 2 are waiting; returns the next round id otherwise; null when all arrived.
  - NUDGE appends `NUDGES[k]` and increments `nudges[id]`; a third NUDGE is ignored.
  - REPLY records `replied[id]` and appends a "me" message; a second REPLY to the same id is ignored; a whitespace-only REPLY is ignored (Review Focus 5).
  - VERDICT for the last unverdicted question sets `phase = "ended"`.
  - LUMBR_ASK increments `asked`; once `asked === MAX_LUMBR_QUESTIONS` further LUMBR_ASK is ignored.
  - LUMBR_ANSWER sets `contextIds` to `answer.contextIds` (server already deduped/capped).
- [ ] **Step 2:** `npm test` → FAIL.
- [ ] **Step 3:** Implement as a pure reducer (no timers; timers live in the component, Task 8). Also record `responseMs[id] = REPLY.at − ARRIVE.at` in state; nothing reads it while `TIME_COUNTS_IN_GRADE` is false. Time-based scoring and pausing the clock during Lumbr calls are deferred until the team decides (spec §3.3).
- [ ] **Step 4:** `npm test` → PASS.
- [ ] **Step 5:** Commit `feat(lumbr-demo): game reducer`.

---

### Task 4: Limits, rate limit, spend store

**Files:** Create `src/lib/lumbr-demo/server/limits.ts`; Test `tests/lumbr-demo/limits.test.ts`. Modify `package.json` (dep `@upstash/redis`).

**Interfaces — Produces:**
```ts
export function validateQuestion(text: unknown): { ok: true; text: string } | { ok: false; reason: "empty" | "too_long" | "type" };
export interface SpendStore { spentTodayUsd(): Promise<number>; addUsd(usd: number): Promise<void> }
export interface RateLimiter { allow(key: string): Promise<boolean> }   // 20 requests / 10 min / key
export function memorySpendStore(): SpendStore;  export function memoryRateLimiter(limit?: number, windowMs?: number): RateLimiter;
export function redisSpendStore(url: string, token: string): SpendStore;   // key lumbr:spend:YYYY-MM-DD (UTC), TTL 2 days
export function redisRateLimiter(url: string, token: string): RateLimiter; // key lumbr:rl:<ip>, TTL = window
export async function liveAllowed(store: SpendStore, budgetUsd: number): Promise<boolean>; // false on store error
```
- [ ] **Step 1: Failing tests.** `validateQuestion("  ")` → empty; 301 chars → too_long; 300 chars → ok, trimmed; non-string → type. Memory rate limiter: 20 allowed then false, allowed again after window (inject clock). `liveAllowed` false when spent ≥ budget; **false when `spentTodayUsd` throws** (Review Focus 2).
- [ ] **Step 2:** `npm test` → FAIL.
- [ ] **Step 3:** Implement (Upstash REST client; memory impls for dev/tests).
- [ ] **Step 4:** `npm test` → PASS.
- [ ] **Step 5:** Commit `feat(lumbr-demo): question limits, rate limit, daily spend ceiling`.

---

### Task 5: Retrieval client, LLM client, citations, ask pipeline

**Files:** Create `server/retrieval.ts`, `server/llm.ts`, `server/ask.ts`, `src/lib/lumbr-demo/citations.ts`, `src/lib/lumbr-demo/match.ts`; Tests `tests/lumbr-demo/citations.test.ts`, `match.test.ts`, `ask.test.ts`

**Interfaces — Consumes:** backend contract (companion plan, "HTTP contract"): `POST /retrieve {query,k}` → `{retriever, top_score, passed, passages}`; `POST /passages {ids}` → `{passages}`; bearer auth. **Produces:**
```ts
export interface RetrievalClient { retrieve(q: string, k: number): Promise<{ topScore: number; passed: boolean; passages: Passage[] }>;
  passages(ids: string[]): Promise<Passage[]>; warm(): Promise<void> }
export function httpRetrieval(baseUrl: string, key: string, timeoutMs: number): RetrievalClient;
export function fakeRetrieval(questions: DemoQuestion[]): RetrievalClient; // keyword overlap over fixture goldSources; passed iff overlap ≥ 2 words
export interface LlmClient { chat(system: string, user: string): Promise<{ text: string; usd: number }> }
export function openAiCompatible(baseUrl: string, apiKey: string, model: string, priceInPerM: number, priceOutPerM: number): LlmClient; // temperature 0
export function fakeLlm(): LlmClient;  // echoes a cited sentence from the first passage, usd 0
export const GENERATOR_SYSTEM: string; export function generatorUser(q: string, passages: Passage[], history: { q: string; a: string }[]): string;
export function numberCitations(raw: string, passages: Passage[]): { text: string; citations: Citation[] };
export function matchWaitingQuestion(q: string, waiting: DemoQuestion[]): DemoQuestion | null; // token Jaccard vs question∪dm, ≥ 0.25, best wins
export type AskInput = { text: unknown; contextIds: unknown; history: unknown; waitingIds: unknown };
export type AskDeps = { retrieval: RetrievalClient; llm: LlmClient; spend: SpendStore; budgetUsd: number; questions: DemoQuestion[] };
export async function askLumbr(input: AskInput, deps: AskDeps): Promise<LumbrAnswer>;
```
`GENERATOR_SYSTEM` = the study's generator prompt (copy it verbatim from the lumbr repo once John's generator track commits it; until then write it from spec §4.4 and cite the spec section in a comment) (answer only from the passages, cite each claim as `[doc_id]`), plus: "If the passages do not contain the answer, reply with exactly: " + `CANNED_LINE`. Passages are rendered as `[doc_id] header\nbody`.

**`askLumbr` order (spec §4, §3.6):**
1. `validateQuestion`; invalid → `kind:"limited"`.
2. `liveAllowed` false → cached match if any (`kind:"cached"`) else `kind:"limited"` with "Lumbr is on a coffee break — try a colleague's question later."
3. Sanitize: `contextIds` → string[] matching `/^dsid_[0-9a-f]+#\d{3}$/`, max `MAX_CONTEXT_PASSAGES`; `history` → last `MAX_HISTORY_TURNS`, each field truncated to 600 chars; anything else discarded.
4. `retrieve(text, 5)` with timeout. On timeout/error → cached match (`kind:"cached"`) else `kind:"waking"` "Lumbr is waking up — try again in a moment." and fire `warm()`.
5. `passed` → context = new passages + `passages(contextIds)`, dedupe by passage_id, newest first, cap 8 → generate.
6. Not passed and `contextIds` non-empty → context = `passages(contextIds)` only → generate.
7. Not passed and no context → `kind:"canned"`, `CANNED_LINE`, **no LLM call**.
8. Generate → `spend.addUsd(usd)`; if text equals `CANNED_LINE` → `kind:"canned"`; else `numberCitations` → `kind:"answer"`, `contextIds` = ids of context passages.

- [ ] **Step 1: Failing tests.**
  - `numberCitations("Shipped Mar 12 [dsid_a#000][dsid_b#001]. Owner Amelia [dsid_a#000].", ps)` → text `"Shipped Mar 12 [1][2]. Owner Amelia [1]."`, 2 citations in first-appearance order; also accepts bare doc ids `[dsid_a]` mapped to that doc's first passage; unknown ids are removed from the text.
  - `matchWaitingQuestion`: a paraphrase sharing key terms matches; an unrelated question → null.
  - `askLumbr` with fakes: below-threshold, no context → canned and `fakeLlm` **not called** (spy); below-threshold with contextIds → LLM called with only those passages; above-threshold → context deduped and ≤ 8; retrieval rejects after timeout → cached when a waiting question matches, else `waking` (Review Focus 1); input with a `passages` field or 50 history turns → prompt contains no forged text and ≤ 6 turns (Review Focus 3); spend store throws → `limited` (Review Focus 2); LLM returning `CANNED_LINE` → `kind:"canned"`.
- [ ] **Step 2:** `npm test` → FAIL.
- [ ] **Step 3:** Implement the five modules.
- [ ] **Step 4:** `npm test` → PASS.
- [ ] **Step 5:** Commit `feat(lumbr-demo): ask pipeline with similarity filter, follow-ups, safety net`.

---

### Task 6: Judge and API routes

**Files:** Create `server/judge.ts`, `server/deps.ts`, `src/app/api/lumbr/ask/route.ts`, `src/app/api/lumbr/judge/route.ts`, `src/app/api/lumbr/warm/route.ts`; Test `tests/lumbr-demo/judge.test.ts`, `tests/lumbr-demo/deps.test.ts`. Create `.env.example`.

**Interfaces — Produces:**
```ts
export const JUDGE_SYSTEM: string;
export async function judgeReply(input: { questionId: unknown; reply: unknown }, deps: AskDeps): Promise<{ verdict: Verdict | "ungraded" }>;
export type RouteDeps = AskDeps & { rateLimiter: RateLimiter };
export function buildDeps(env: NodeJS.ProcessEnv): RouteDeps; // fakes when LUMBR_BACKEND_URL / LLM_API_KEY unset AND NODE_ENV !== "production"
```
Judge prompt gets the colleague question, `answerFacts`, `goldAnswer`, the reply; returns JSON `{"verdict": "correct"|"partial"|"wrong"}`. For `infoNotFound` questions the prompt states an honest "I couldn't find it / don't know" is **correct** and any invented specific answer is **wrong**. Routes: `POST /api/lumbr/ask` (body `AskInput`, IP from `x-forwarded-for` first hop → rate limiter → `askLumbr`), `POST /api/lumbr/judge`, `GET /api/lumbr/warm` (calls `retrieval.warm()`, returns 204). 429 when rate-limited.
Env (`.env.example`): `LUMBR_BACKEND_URL`, `LUMBR_BACKEND_KEY`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_PRICE_IN_PER_M`, `LLM_PRICE_OUT_PER_M`, `LUMBR_DAILY_BUDGET_USD=0.33`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`.

- [ ] **Step 1: Failing tests.** `judgeReply` parses `{"verdict":"partial"}` and fenced/prose-wrapped JSON; unparseable → `ungraded`; LLM throws → `ungraded`; unknown questionId → `ungraded`; whitespace reply → `wrong` without an LLM call (Review Focus 5). `buildDeps` in production with missing Upstash env → spend store whose `spentTodayUsd` rejects (so `liveAllowed` is false).
- [ ] **Step 2:** `npm test` → FAIL.
- [ ] **Step 3:** Implement; routes are thin wrappers (parse JSON, call, `Response.json`).
- [ ] **Step 4:** `npm test` → PASS; `npm run dev`, then `curl -s -XPOST localhost:3000/api/lumbr/ask -d '{"text":"did we ship the idempotency fix?","contextIds":[],"history":[],"waitingIds":[]}'` returns a fake `answer` with citations, and `-d '{"text":"write me a poem",…}'` returns `canned`.
- [ ] **Step 5:** Commit `feat(lumbr-demo): judge and API routes`.

---

### Task 7: Page, window chrome, Lumbr icon, desktop shell

**Files:** Create `src/app/projects/[slug]/demo/page.tsx`, `src/components/lumbr-demo/{window-frame,taskbar,toast,desktop-icons,lumbr-log-icon,desktop,about-window}.tsx`, `src/data/lumbr-demo/{easter-eggs,about}.ts`. Modify `content/projects/lumbr-enterprise-rag.mdx` frontmatter.

**Interfaces — Produces:** `<WindowFrame title onClose? icon? className>`; `<LumbrLogIcon size waiting?: boolean>` (leaf wiggles via CSS keyframes when `waiting`, `prefers-reduced-motion` disables); `<Desktop>` slots for windows/toasts; `easterEggs: { label: string; glyph: string; title: string; body: string }[]` (4 placeholders, e.g. "Recycle Bin (full)", "Q3_OKRs_FINAL_v2(3).xlsx", "Onboarding (empty)", one more); `about: { retriever: string; studyScore: string; differences: string[] }` (placeholder until the study picks a retriever).

- [ ] **Step 1:** Page: `generateStaticParams` returns only `{ slug: "lumbr-enterprise-rag" }`; any other slug → `notFound()`. Renders a client `<DemoApp>` (Task 8) inside `fixed inset-0 z-[100]` so it covers nav/footer; metadata title "First Day at Redwood".
- [ ] **Step 2:** Desktop visuals: CSS/SVG sky gradient + original rolling hill, icons grid (Lumbr, Redwood Messenger, easter eggs), taskbar with "redwood" start button (opens a menu with "About this demo" → `<AboutWindow>` and "Log off" → back to project page), clock, task buttons. Match the approved mockup (`lumbr/.superpowers/brainstorm/*/content/desktop-v2.html`).
- [ ] **Step 3:** Frontmatter: `demoUrl: "/projects/lumbr-enterprise-rag/demo"`, `demoLabel: "Try your first day"`.
- [ ] **Step 4:** Verify: `npm run build` succeeds and lists the demo route; `npm run lint` clean; open `/projects/lumbr-enterprise-rag/demo` in the browser — desktop renders full-screen, no site nav visible; `/projects/diabetes-prediction/demo` is 404.
- [ ] **Step 5:** Commit `feat(lumbr-demo): demo page, retro desktop shell, Lumbr icon`.

---

### Task 8: Playable round (desktop)

**Files:** Create `src/components/lumbr-demo/{demo-app,messenger,lumbr-chat,citation-popover,ending-document}.tsx`

**Interfaces — Consumes:** `gameReducer`, `nextArrival`, `waitingIds` (Task 3), `pickRound` (Task 2), `/api/lumbr/{ask,judge,warm}` (Task 6), chrome (Task 7).

- [ ] **Step 1:** `DemoApp`: `useReducer(gameReducer)`; on mount `pickRound(questions, ROUND_LENGTH, Math.random)` and `fetch('/api/lumbr/warm')`; welcome window with `WELCOME_TEXT`; START on dismiss or after showing, first ARRIVE after `FIRST_COLLEAGUE_DELAY_MS`; then schedule ARRIVE whenever `nextArrival` is non-null after a random delay in `NEXT_COLLEAGUE_DELAY_MS`; schedule NUDGEs at `NUDGE_AFTER_MS` per waiting colleague; clear timers on unmount; pause timers when `document.hidden`.
- [ ] **Step 2:** Toasts from the taskbar for each ARRIVE/NUDGE (click opens Messenger on that thread). Messenger: contact list (unread dot), thread, reply box → REPLY then `POST /api/lumbr/judge` → VERDICT → colleague reaction line by verdict ("perfect, thanks!" / "ok… but that's not the whole story" / "hm, that's not what I heard"; ungraded → "thanks!").
- [ ] **Step 3:** Lumbr window: input (maxLength 300, counter), LUMBR_ASK → `POST /api/lumbr/ask` with `{ text, contextIds, history, waitingIds }` → LUMBR_ANSWER; "Lumbr is thinking…" state; numbered citation chips open `<CitationPopover>` (header + body); `cached` answers show a small "cached" tooltip. Lumbr icon `waiting` = `waitingIds(state).length > 0`.
- [ ] **Step 4:** Ending: when `phase === "ended"`, `<EndingDocument>` from `scoreRound`: pass → "First-Day Performance Review", fail → "Performance Improvement Plan", null → "Ungraded" variant; table of every question: colleague, their DM, visitor reply, verdict, `goldAnswer`, `goldSources[0].header`; "Start another first day" resets with a new `pickRound`.
- [ ] **Step 5:** Verify in the browser with fakes (dev server): play a full round of 6; confirm welcome → first toast at ~3 s, ≤ 2 colleagues waiting, nudges appear, citations open, an off-topic Lumbr question returns the canned line, ending shows the right document; a second round has different questions.
- [ ] **Step 6:** Commit `feat(lumbr-demo): playable desktop round`.

---

### Task 9: Phone layout

**Files:** Create `src/components/lumbr-demo/mobile-shell.tsx`; Modify `demo-app.tsx`

- [ ] **Step 1:** `DemoApp` picks `<MobileShell>` when `matchMedia('(max-width: 767px)')` matches (listen for changes; same reducer state, so rotating mid-round keeps progress).
- [ ] **Step 2:** `MobileShell`: one full-screen chat list (Lumbr pinned first as a contact, colleagues with unread badges), tap → thread view with back button; Lumbr thread is the Lumbr chat; ARRIVE/NUDGE show a top banner (auto-hide 4 s, tap opens thread); welcome and ending render full-screen; no desktop icons or easter eggs; retro styling (title bar, Tahoma-like font, faded wallpaper behind the list).
- [ ] **Step 3:** Verify at 375×812 in the browser: full round playable with fakes, no horizontal scroll, input not hidden by the keyboard area (use `dvh` units).
- [ ] **Step 4:** Commit `feat(lumbr-demo): phone layout`.

---

### Task 10: Precompute cached answers and go live

**Blocked on:** companion plan Tasks B4–B5 (real `questions.json`, deployed backend).

**Files:** Create `scripts/precompute-lumbr-answers.ts`; Replace `src/data/lumbr-demo/questions.json`; Modify `src/data/lumbr-demo/about.ts`

- [ ] **Step 1:** Script: reads `questions.json`, runs `askLumbr` (real deps from env, empty context/history) on each `question`, writes `cached` (`kind` answer or canned) back; exits non-zero if `LUMBR_BACKEND_URL` or `LLM_API_KEY` is unset, or if any id starts with `fake-`. Run: `npx tsx scripts/precompute-lumbr-answers.ts` → every entry has non-null `cached`; INF entries are canned.
- [ ] **Step 2:** Fill `about.ts` with the deployed retriever, its study Recall@10, and differences (int8 index, rerank depth if any, multi-turn follow-ups not measured by the study).
- [ ] **Step 3:** Provision Upstash Redis from the Vercel Marketplace on the site project; set all `.env.example` vars in Vercel (Production + Preview). Set a hard spend cap at the LLM provider too.
- [ ] **Step 4:** `npm test && npm run build`; deploy a preview; play a round on the preview URL (desktop + phone); confirm `/api/lumbr/ask` with an off-topic question returns canned in < 2 s and that the Upstash `lumbr:spend:<today>` key grows after a real answer.
- [ ] **Step 5:** Commit `feat(lumbr-demo): real questions, cached answers, about panel`; open PR to `main`.
