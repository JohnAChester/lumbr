# "First Day at Redwood": demo design

**Date:** 2026-11-16 · **Owner:** John · **Status:** design, pending John's approval
**Lives on:** John's personal site (`personal_site`, Next.js 16.4 on Vercel), linked from the existing Lumbr
project page (`content/projects/lumbr-enterprise-rag.mdx`).

Decisions are tagged **[John]** (decided by John), **[Prof]** (recommended by John's professor and adopted), or
**[proposal]** (already promised in the course proposal, `Group10-Lumbr_RAG_Assistant.docx`). Untagged
specifics marked *(Claude's call)* were proposed during brainstorming and not objected to. John can overturn them.

---

## 1. Intent

Lumbr is a CISC 6000 group project: a cited-answer RAG assistant over EnterpriseRAG-Bench (512K synthetic
documents from the fictional company Redwood Inference), plus a study comparing six retrievers (R1–R6). The
proposal promises a playable demo, "First Day at Redwood," deployed on a team member's site. Success
criteria in the proposal include "a working Lumbr with citations … powering a playable 'First Day at Redwood'
simulation."

The demo turns the benchmark into something a visitor can feel. It's your first day, colleagues are asking
about things you know nothing about, and Lumbr, with citations, is the only way to answer. Its job is to
show why retrieval-grounded, cited, abstaining answers matter. It also needs to be fun enough that a
portfolio visitor plays a full round.

**Why now:** a Google Drive OAuth outage blocked Colab work, and most of the demo doesn't depend on Colab.
Drive is reachable again, so the v0 passage sample is available.

## 2. Goals, non-goals, anti-goals

**Goals**
- A complete, replayable round: about 6 colleague questions, then a letter grade and an HR-document ending.
- Real retrieval and generation on questions the visitor types, with clickable citations.
- Honest abstention: Lumbr says it can't find something rather than inventing an answer.
- Runs for months on about **$10/month** **[John]**, and fails gracefully when the backend is down or over budget.

**Non-goals**
- Reproducing the study's exact numbers in the demo (CPU-only hosting, compressed index; see §6).
- Accounts, leaderboards, or storing anything about visitors.
- A general-purpose chatbot.

**Anti-goals (it fails if…)**
- An off-topic or abusive input costs real money, or anyone can run up the bill.
- Lumbr confidently answers with no support from the documents.
- A visitor sits through a hang with no feedback (cold start, API outage).
- The page copies Microsoft's real assets (see §3.1).
- Benchmark **test** questions are used anywhere in the demo, including threshold tuning (study data rule).

## 3. The experience

### 3.1 Setting: a retro desktop [John]
- The whole demo is **one full-screen page** of its own, not embedded in the write-up **[John]**. The project
  page gets a "Try your first day" button. Route *(Claude's call)*: `/projects/lumbr-enterprise-rag/demo`.
- The screen is a **retro, early-2000s desktop** **[John]**: a rolling green hill under a blue sky as the
  wallpaper, desktop icons, a taskbar with a start button and clock, and windows with period-style title bars.
- **It's an original homage, not a copy** *(Claude's call, John didn't object)*: hand-drawn or CSS/SVG wallpaper,
  a "redwood" start button, no Microsoft logos and no real Bliss photo. Using those on a public site invites a
  takedown.
- **Funny easter eggs as desktop icons** **[John]**. Examples from the mockup: "Recycle Bin (full)",
  "Q3_OKRs_FINAL_v2(3).xlsx", "Onboarding (empty)". **The final list and their gags are John's to write.** The
  builder provides the mechanism (an icon opens a small window or dialog with static content), plus 3–4
  placeholders.

### 3.2 Flow of a round
1. **Welcome.** The opening message from the proposal (verbatim, below) appears in a window.
   > "Welcome to your first day at Redwood Inference! We're so excited to have you on the team. We're
   > fast-paced, so all our new employees must hit the ground running. If you have any questions, Lumbr is here
   > to help. We expect great things from you!"
2. **After 3 seconds** **[proposal]**, colleague messages start **popping up from the bottom of the screen**
   (taskbar-style notifications) **[John]**. Colleagues are named after people who appear in the corpus
   **[proposal]**. Clicking a notification opens the **messenger window** (contact list on the left, the
   conversation on the right, a reply box).
3. **Lumbr is an icon on the desktop** **[John]**, drawn as a **fat, stylized log tilted 45° with a small leaf off
   the side** (flat app-icon style, not realistic) **[John]**. Opening it opens a Lumbr chat window. The icon
   draws attention while a colleague is waiting (e.g. the leaf wiggles) *(Claude's call)*. Inline SVG so it scales.
4. **The visitor types their own question to Lumbr** **[John]**. Nothing is pre-filled or copied over. Lumbr
   answers with **numbered citations**. Clicking one shows the source passage's header (source, channel,
   author, date) and body **[proposal]**.
5. The visitor **types a reply to the colleague** in the messenger.
6. **The reply is graded** (§3.3), and the colleague reacts in character: "perfect, thanks!" / "ok… but what
   about the telemetry piece?" / "hm, that's not what Jira says" *(wording: Claude's call)*.
7. **Colleagues get impatient** if the visitor is slow **[John]**: follow-up nudges ("??", "hello?? standup in
   5"). Several colleagues can be waiting at once, staggered.
8. After the last question: **the ending screen** (§3.4).

### 3.3 Grading
- **A small LLM judge** **[John]**, using the same pay-per-token model as the generator, gets the colleague's
  question, the benchmark's **answer facts** for it, and the visitor's reply. It returns **correct / partly right /
  wrong**. ROUGE was considered and rejected: it rewards copied wording and punishes correct paraphrases.
- **About 6 questions per round** **[John: 5–10]**, drawn at random from the **100-question dev split**, mixed
  across benchmark categories so a round is neither all trivial lookups nor all multi-document questions
  *(Claude's call)*.
- **One "Info Not Found" trap per round** *(Claude's call, John liked it)*: Lumbr can't find it, and the right reply
  is an honest "I couldn't find that." The judge scores honest non-answers as correct on these questions.
- **Score:** correct 1, partly ½, wrong 0, averaged. A ≥ 90%, B ≥ 75%, C ≥ 60%, D ≥ 45%, F below that.
  *(Claude's call, John liked it)*
- **Response time:** whether it affects the grade is **open. John is deciding with Patricia and Jieun.** Build it
  behind one config switch (default: time does not count). If time does count, **pause the clock while Lumbr is
  working**, so cold starts and API latency never cost the visitor.
- If the judge is unavailable (backend down, budget hit), the reply counts as ungraded, the colleague just says
  thanks, and the ending grades only what was judged.

### 3.4 Ending [John]
A mock HR document *(format: Claude's call, John liked it)*:
- **Pass (A–C):** a cheerful "First-Day Performance Review."
- **Fail (D/F): a "Performance Improvement Plan."** **[John: "if they fail, we PIP them"]**
- Both list every question with the visitor's verdict, **the correct answer, and its cited source**. This is
  where visitors see what grounded answers buy them. Both end with "Start another first day."

### 3.5 Phones [John]
On narrow screens there is no desktop. **It's just chat windows, and Lumbr is an account you can message**,
listed alongside the colleagues in the conversation list. Same round, grading, nudges and ending. Details
*(Claude's call)*: new colleague messages appear as in-app banners at the top, and the ending document is shown
full-screen. Desktop easter eggs are left out on phones. The retro look carries through as the chat styling
(title bars, fonts, wallpaper behind the list). The breakpoint is the builder's choice, matched to the site's
existing breakpoints.

### 3.6 Follow-up questions to Lumbr [John]
Lumbr keeps a conversation. **Follow-ups are answered by the generator using the documents already pulled**
for the conversation, along with the recent conversation history. How that combines with the similarity
filter *(Claude's call, flagged for John)*:
- Every message still runs retrieval. If it passes the threshold, the generator gets the new passages **plus**
  the passages already pulled earlier in the conversation, deduplicated and capped (e.g. the 8 most recent).
- If a message **fails** the threshold but the conversation already has pulled passages (typical of "who wrote
  that?"), it goes to the generator with **only the earlier passages**. The generator must answer from them or
  reply with the exact canned line from §4.2.
- If it fails and there's no prior context, the canned line comes back immediately and the generator isn't called (unchanged).
- Cost risk: after one good question, junk gets one cheap generator call per message instead of zero. The
  per-round cap and daily ceiling bound this. Citations in follow-up answers still point to the passages used.
- The demo's multi-turn behavior goes beyond what the study measured. The demo page says so in one line.

## 4. What happens to a typed question

Layers run cheapest-first. A question only reaches the paid model if every cheaper layer lets it through.

1. **Hard limits** *(Claude's call, John agreed)*: a maximum question length (about 300 characters), a cap on
   Lumbr questions per round (about 30), and a **daily spend ceiling** past which live mode turns off with an in-world
   message ("Lumbr is on a coffee break"). Rate limit per IP as well.
2. **Similarity filter** **[Prof]**: retrieve first. If the top score is below a threshold, return exactly
   *"I'm sorry, but I can't find that information in our database."* and **do not call the generator.**
   - Compare the **dense cosine similarity** (or the reranker score if a reranker is served), never BM25 or RRF
     scores, which aren't on a comparable scale *(Claude's call)*.
   - **Tune the threshold on dev questions only**: answerable vs. "Info Not Found", plus a small hand-written
     off-topic set. Pick the cutoff that blocks junk while rarely blocking real questions. Record the result,
     since it may go in the report.
3. **No gate model** **[John]**. The tiny "is this a workplace question?" LLM call was considered and dropped.
   Revisit only if testing shows junk getting past layer 2.
4. **Generator** **[proposal]**: the study's fixed open 7–8B instruct model (Llama 3.1 8B or Qwen2.5 7B,
   whichever the study fixes) via a **pay-per-token API**, temperature 0, with the study's prompt: answer only
   from the top 5 passages, cite each claim by doc ID, or say the information isn't available. Rendered with
   numbered citations in the UI.
5. **Safety net** *(Claude's call)*: precomputed Lumbr answers for all 100 dev questions, generated offline with
   the same pipeline. If the live backend is cold, slow (past a few seconds) or off, and the visitor's question is
   close to **any waiting colleague's** question, serve the stored answer, marked as cached in a small tooltip
   *(Claude's call)*. "Close" can be decided in the browser or the Next.js route. The matching method is left to
   the builder.
6. **Round length** is a config value, default 6 (range 5–10 **[John]**).

## 5. Architecture and hosting

```
Browser (retro desktop, React)
   │  questions, replies
   ▼
personal_site  ── Next.js API routes (Node): limits, spend counter, calls to the backend and the LLM API
   │                                         │
   ▼                                         ▼
Retrieval backend (Python)              Pay-per-token LLM API
 query embedding + index search            generator + judge
 + similarity filter
```

- **Retrieval backend: a Vercel Python function in its own Vercel project** *(Claude's call after checking current
  limits, John agreed)*. Hobby functions get **2 GB RAM and 1 vCPU**. Python bundles can be 500 MB, or up to
  5 GB with the large-functions beta. So:
  - Serve the dense retriever on CPU (ONNX or Sentence-Transformers), with the index **compressed to int8**
    (about 500 MB for 640K × 768 instead of about 2 GB). This should lose roughly 1% recall; measure it on dev.
  - **Fallback: a free Hugging Face Space** (about 16 GB RAM, sleeps when idle, 30–60 s wake), if 2 GB is too tight,
    or if the reranker must be served.
  - Code lives in the **lumbr repo** (e.g. `demo/backend/`) so it can reuse `lumbr.schema` and the index
    artifacts *(Claude's call)*. Model and index files are pulled at build or cold start from Drive exports or
    the HF Hub, not committed to git.
  - Assumption to verify: John's site is on the Hobby plan.
- **Cross-encoder reranker [John, 2026-11-16]:** the team is building a cross-encoder reranker on document retrieval (the study's R6). Building it is the team's work, not the demo's; the demo only serves the finished reranker. That makes the reranker score the filter's score (§4.2), and memory the main hosting risk: the reranker plus the index may not fit in 2 GB, which would trigger the HF Space fallback.
- **Which retriever** *(Claude's call)*: retrieval is a setting, as in the study. At deploy time, serve **the best
  study setup that runs well on CPU**. R1, R2, R4 and R5 are feasible. R3 (4B embedder) is not. R6 is feasible
  only with a short rerank list or on the Space fallback. **The demo page states which setup it runs, its
  study score, and how it differs from the study** (int8 index, rerank depth).
- **Corpus:** the study's subsample (~140K docs, ~640K passages, `processed/passages.jsonl`) once it exists.
  Develop against the **v0 sample** (`processed/sample_passages.jsonl`, 4.6K passages, contains every gold doc
  of 50 questions).
- **Site side:** new route(s) under `src/app/api/` (the site already has `api/citibike`), plus the page and its
  components. Node runtime. Don't use Edge (unsupported on Next.js 16.3+). Read `node_modules/next/dist/docs/`
  first, as `personal_site/AGENTS.md` requires.
- **Spend tracking:** the daily ceiling needs a counter that survives across function instances (e.g. a small
  Redis/KV from the Vercel Marketplace, or the LLM provider's own spend cap). The mechanism is left to the builder,
  but it must hold under concurrent visitors.
- **Privacy:** don't store visitors' questions or replies beyond what's needed to serve the request. No visitor
  identifiers are kept.

**Cost picture (estimates):** generator and judge are about 2K input tokens per call at roughly
$0.10–0.20 per million tokens, so a few hundredths of a cent each. Retrieval is free or cents on the Vercel
free tier. $10/month covers tens of thousands of questions.

## 6. Data the demo needs (and where it comes from)

| Artifact | Source | Ready? |
|---|---|---|
| Dev split (100 question IDs) | Study milestone "dev/test split fixed" | **Not in the repo yet.** Confirm with Patricia. |
| Per question: text, category, gold answer, **answer facts**, gold doc IDs | EnterpriseRAG-Bench (HF release) | Yes |
| Colleague names (people from the corpus) | Corpus `author` fields. A colleague can be someone who wrote one of the gold docs, or any frequent author. | Builder decides |
| Colleague phrasing of each question | Rewrite the benchmark question into a casual DM (by hand or LLM, reviewed) | Builder |
| Precomputed answers (safety net) | Run the demo pipeline over the dev split | After the retriever is chosen |
| Threshold | Dev split + off-topic set (§4.2) | After the retriever is chosen |
| Index + model files | Study outputs (Patricia: indexes; Jieun: R4 model) | About Nov 23 |

## 7. Build order (works around the Colab and study timeline)

1. **Desktop UI and game loop** on the site with a handful of hand-written fake questions, answers and facts.
   Needs nothing from Drive.
2. **Backend** on the v0 sample with R2 (base model, no fine-tuning). Deploy it and measure memory, cold start and latency.
3. **LLM wiring:** generator and judge through the API, limits, spend ceiling.
4. **Real content:** dev questions, colleague rewrites, threshold tuning, precomputed answers.
5. **Swap in the winning feasible retriever and full index** (about Nov 23–27), then the "what this demo runs" note.

The proposal targets the deploy for Nov 24–27; the course deadline is **Nov 30**.

## 8. Open questions (John's)

- Does response time affect the grade? (John, with Patricia and Jieun; the switch is built either way.)
- The easter-egg list and gags (the builder ships placeholders).
- Dev split: needs to come from the study (Patricia) before step 4 of the build order. Not a design question,
  but it blocks real content.

## 9. Left to the builder

Visual polish of the desktop and windows; animation details; exact colleague reactions and nudge timing;
question-mixing logic; how "close to the current question" is computed for the safety net; ONNX vs.
Sentence-Transformers on CPU; index format (FAISS int8, numpy, etc.); the LLM provider (any pay-per-token host of
the study's generator); the spend-counter mechanism; component structure.
