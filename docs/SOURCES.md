# Source notes: EnterpriseRAG-Bench (issue #2)

**Source of truth:** the structured JSON in [`onyx-dot-app/EnterpriseRAG-Bench`](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
at commit `d36685e2` (pinned as `sources.GH_SHA`). `sources.build_json_parquet()` streams the repo tarball
into one file, `lumbr-data/raw/erb_json.parquet` (1.07 GB, `doc_id, source_type, path, json`), in about 2 minutes.
The Hugging Face release (`data.download_raw()`) is used only for cross-checks.

## Findings

| Check | Result |
|---|---|
| Doc IDs, GitHub JSON vs HF | **Identical sets**: 511,958 unique IDs in each, 0 missing either way, 0 source mismatches. Titles match 180/180 sampled. |
| Body text, GitHub JSON vs HF `content` | 96–100% identical (difflib, 20 docs per source). The differences are an **HF artifact**: HF stores list fields as JSON strings with escaped `\n`, so the GitHub JSON is the cleaner text. |
| Metadata (HF has none) | channel/author/date keys are **100% filled** in every source (see the mapping below). |
| Doc ID collisions | **4 doc IDs are each shared by two different files** (e.g. a Confluence page and a HubSpot record), identically in HF. The chunker treats both files as one doc (chunks numbered continuously), so either counts for that gold ID. See `sources.duplicate_ids()`. |
| Content fields vary within a source | 5 to 1,500+ `content_field_names` variants per source. Rendering always follows each doc's own `content_field_names`, never hard-coded keys. |
| Literal `\n` escapes | 2–30% of docs per source contain the two characters `\` `n` instead of newlines. The chunker unescapes them (`chunk.clean_text`). |
| Bogus Slack timestamps | **47% of Slack threads** have epoch values that decode to 2038+ (made-up by the generator). Dates outside 2020–2029 become `null`, never a wrong date. |
| Gmail quoted replies | 16% of emails quote the previous email in full (`On … wrote:` / `>` lines). These are stripped, since the original is already its own unit. |
| Gmail formats | 95% `messages: list[str]` (one email each, starting `From:/To:/Date:/Subject:`). The rest are single strings, sometimes with `-----MESSAGE n-----` / `--- Message n of m ---` separators. All are handled. |
| Long "titles" | A few docs point `title_field_name` at a long text field (up to 8.7K chars). Titles are cut to 150 chars. |
| Slack speaker names | Often lowercase (`zoe:`, `rafa:`, `infra-bot:`). Speaker detection allows that. |

## Size per source (from chunking 2,000 docs per source)

| source | docs | % corpus | passage tokens / doc | passages / doc |
|---|---|---|---|---|
| slack | 285,605 | 55.8% | 993 | 3.6 |
| gmail | 121,390 | 23.7% | 1,951 | 5.5 |
| linear | 35,308 | 6.9% | 1,436 | 5.0 |
| jira | 6,120 | 1.2% | 1,650 | 5.7 |
| hubspot | 15,017 | 2.9% | 906 | 3.2 |
| google_drive | 25,108 | 4.9% | 2,090 | 7.6 |
| fireflies | 10,173 | 2.0% | 3,377 | 10.8 |
| github | 8,052 | 1.6% | 1,303 | 4.4 |
| confluence | 5,189 | 1.0% | 2,875 | 10.3 |

Corpus-weighted: **about 4.6 passages per doc**. That makes the full corpus about 2.3M passages, and the ~140K-doc subsample (#5) about 640K passages,
or about 190M tokens to embed per dense model.

## Metadata mapping (D6/D7, implemented in `chunk.META_KEYS`)

| source | channel | author | date (most recent activity) | created |
|---|---|---|---|---|
| slack | `channel` | first speaker in the thread | `last_message_ts` (epoch; null if implausible) | `first_message_ts` |
| gmail | `mailbox_owner` | first email's `From:` name | `last_email_at` | `first_email_at` |
| linear | `project` (else `team`) | `creator` | `updated_at` | `created_at` |
| jira | `project` | `reporter` | `updated_at` | `created_at` |
| hubspot | `stage` | `owner` | `updated_at` | `created_at` |
| google_drive | `team` (else `drive_area`) | `owner` | `last_modified` | `created_at` |
| fireflies | `call_type` | `redwood_owner` | `recorded_at` | `recorded_at` |
| github | `repo` | `author` | `updated_at` | `created_at` |
| confluence | `space` | `author` | `last_updated` | `created_at` |

Missing values are `null`, never guessed. In a 112K-passage run: author is null for 0.1% (mostly Gmail threads
whose first email has no `From:` line), and date is null for 46% of Slack passages (the bogus epochs).

## Per-source key survey (keys with ≥ 10% fill; 2,000 docs per source)

Regenerate with `survey.markdown_report(survey.sample_by_source(2000), min_fill=0.10)`.

#### slack

285,605 docs; surveyed 2,000. Title field: {'channel': 1934, 'thread_body_preview': 1, 'original_location': 12, 'file_name': 11, 'note': 1, 'thread_ts': 15, 'file_path': 6, 'thread': 3, 'thread_summary': 3, 'root_message': 1, 'thread_filename': 1, 'file_notes': 1, 'filename': 1, 'summary': 2, '_filename': 1, '_original_filename': 1, 'thread_text_preview': 1, 'file_note': 1, 'filename_hint': 1, 'dataset_doc_uuid': 1, 'filename_note': 1, 'thread_text': 1}.

Content-field variants (most common):

- 44% `['messages']`
- 39% `['text']`
- 13% `['thread']`
- 3% `['content']`
- 0% `['thread_text']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `channel` | 100% | str |  | random |
| `thread_ts` | 100% | str |  | 2401234567 |
| `first_message_ts` | 100% | str |  | 2401234567 |
| `last_message_ts` | 100% | str |  | 2401234870 |
| `participants` | 100% | list[str] |  | zoe |
| `messages` | 44% | str | yes | alex_infra: Heads up — scheduling a maintenance window for the us-west… |
| `text` | 39% | str | yes | zoe: Huge shoutout to @rafa and @kyle for swooping in last night when … |
| `thread` | 13% | str | yes | incident-bot: PAGED [sev=high] decode-worker-12 GPU OOM spikes detecte… |

#### gmail

121,390 docs; surveyed 2,000. Title field: {'subject': 2000}.

Content-field variants (most common):

- 95% `['messages']`
- 1% `['body']`
- 1% `['thread']`
- 0% `['content']`
- 0% `['emails']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `thread_id` | 100% | str |  | soojin-20261119-nightwindow-01 |
| `mailbox_owner` | 100% | str |  | soojin_lee |
| `subject` | 100% | str |  | Night Window: Private Deploy Canary & Smoke Acceptance Plan |
| `participants_internal` | 100% | list[str] |  | soojin_lee |
| `message_count` | 100% | str |  | 5 |
| `first_email_at` | 100% | str |  | 2026-11-19T02:14:00-08:00 |
| `last_email_at` | 100% | str |  | 2026-11-19T10:37:00-08:00 |
| `thread_type` | 100% | str |  | misc |
| `has_attachments` | 100% | str |  | true |
| `attachments` | 100% | list[str] |  | night-window-checklist.xlsx |
| `participants_external` | 100% | list[str] |  | ACME SRE <sre@acme-corp.com> |
| `related_links` | 99% | list[str] |  | https://drive.google.com/drive/folders/1-example-night-window |
| `related_account` | 99% | str |  | ACME Corp |
| `messages` | 96% | list[str] | yes | From: soojin_lee@redwood.com To: jonas_weber@redwood.com, vivek_kulkar… |
| `region` | 94% | str |  | us |
| `deal_id` | 79% | str |  | DW-4982 |

#### linear

35,308 docs; surveyed 2,000. Title field: {'title': 2000}.

Content-field variants (most common):

- 13% `['description', 'comments']`
- 4% `['description', 'acceptance_criteria', 'comments']`
- 2% `['description', 'acceptance_criteria', 'tasks', 'comments']`
- 1% `['description', 'acceptance_criteria', 'notes', 'comments']`
- 1% `['description', 'acceptance_criteria', 'comments', 'notes']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `key` | 100% | str |  | ENG-19708 |
| `team` | 100% | str |  | engineering |
| `title` | 100% | str |  | SLO-aware prefill shaper and kernel hinting to cut decode tail latency |
| `status` | 100% | str |  | In Progress |
| `priority` | 100% | str |  | P0 |
| `created_at` | 100% | str |  | 2026-02-20 |
| `updated_at` | 100% | str |  | 2026-03-13 |
| `creator` | 100% | str |  | Nora Patel |
| `assignee` | 100% | str |  | Diego Ramirez |
| `estimate` | 100% | str |  | 8 |
| `labels` | 100% | list[str] |  | batching |
| `project` | 100% | str |  | runtime-performance |
| `cycle` | 100% | str |  | 2026-W09 |
| `description` | 98% | str | yes | Problem: Several production customers are experiencing occasional 95th… |
| `release` | 98% | str |  | runtime-1.20 |
| `security_review_required` | 96% | str |  | false |
| `due_date` | 96% | str |  | 2026-03-22 |
| `customer_impact` | 94% | str | some | high |
| `dependencies` | 94% | list[str] | some | ENG-8421 |
| `comments` | 76% | list[str] | yes | 2026-02-22 - Nora Patel: Initial triage and root-cause summary from pr… |
| `acceptance_criteria` | 75% | list[str] | yes | 95th percentile decode latency reduced by >=25% on mixed workloads in … |
| `parent_issue` | 49% | list[str] |  | ENG-76100 |
| `links` | 46% | list[str] | some | https://confluence.redwood.ai/runtime/prefill-shaping-proposal |
| `attachments` | 26% | list[str] | some | design-diagram-eviction-state.png |
| `notes` | 25% | list[str] | yes | This ticket focuses on the 'locked' model path used by customers who p… |
| `rollout_plan` | 22% | list[str] | yes | Stage 1 - Lab: run replay benchmarks with synthetic mixed prefill+deco… |
| `tasks` | 15% | list[str] | yes | Create entitlement mapping spreadsheet and owner sign-off - https://dr… |
| `next_steps` | 15% | list[str] | yes | Implement tokenizer parity module and integrate into CI (owner: Liam). |
| `implementation_notes` | 12% | list[str] | yes | - Add request hydrator middleware in compat routing layer that runs pr… |
| `risks` | 11% | list[str] | yes | Over-shaping could starve high-throughput batch workloads and increase… |
| `updates` | 10% | list[str] | yes | 2026-02-08 - 14:27 UTC: Incident detected via streaming-error-rate ale… |

#### jira

6,120 docs; surveyed 2,000. Title field: {'summary': 1998, 'summary_short': 2}.

Content-field variants (most common):

- 6% `['description', 'comments']`
- 6% `['description', 'comments', 'investigation_notes', 'resolution']`
- 5% `['description', 'comments', 'investigation_notes', 'resolution_notes']`
- 4% `['description', 'steps_to_reproduce', 'logs', 'investigation_notes', 'comments', 'resolution']`
- 4% `['description', 'comments', 'resolution']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `key` | 100% | str |  | INT-1000423 |
| `project` | 100% | str |  | internal-support |
| `issue_type` | 100% | str |  | Task |
| `summary` | 100% | str |  | Tag-driven project lifecycle automation: decommission Slack channels, … |
| `status` | 100% | str |  | In Progress |
| `priority` | 100% | str |  | P2 |
| `created_at` | 100% | str |  | 2026-03-14 |
| `updated_at` | 100% | str |  | 2026-03-15 |
| `reporter` | 100% | str |  | Jordan Kim |
| `assignee` | 100% | str |  | Priya Desai |
| `components` | 100% | list[str] |  | slack |
| `labels` | 100% | list[str] |  | lifecycle-automation |
| `description` | 99% | str | yes | Issue summary: We need a tag-driven project lifecycle automation that … |
| `environment` | 99% | str | some | prod |
| `related_confluence_pages` | 99% | list[str] |  | https://confluence.redwood.ai/display/PLAT/Tag-driven+Lifecycle+Automa… |
| `severity` | 98% | str |  | Sev2 |
| `linked_issues` | 98% | list[str] |  | INT-572801 |
| `comments` | 98% | list[str] | yes | 2026-03-14 Jordan Kim: Filing request on behalf of Legal & Platform. P… |
| `related_github_prs` | 96% | list[str] |  | https://github.com/redwood-inference/platform/pull/3482 |
| `affected_regions` | 81% | list[str] |  | us-west |
| `investigation_notes` | 80% | list[str] | yes | - Checked vendor master: vendor NorthStar Components (vendor_id V-8821… |
| `sla_due_at` | 76% | str |  | 2026-03-14 |
| `customer_tier` | 74% | str |  | dedicated |
| `customer_company` | 74% | str |  | ParcelFlow.ai |
| `affected_models` | 67% | list[str] |  | gpt-medium-open-2025 |
| `resolution` | 64% | str | yes | Planned - design reviewed. Implementation sprint scheduled (Platform E… |
| `first_response_due_at` | 57% | str |  | 2026-03-13 |
| `steps_to_reproduce` | 46% | list[str] | yes | 1) Upload a single invoice with mixed PO lines across entities to AP i… |
| `logs` | 35% | list[str] | yes | 2026-03-09T21:12:44Z - match-service INFO - Invoice INV-2026-0331 matc… |
| `resolution_notes` | 26% | str | yes | - 2026-03-14: Payments team processed manual JE and paid INV-2026-0331… |
| `next_steps` | 19% | list[str] | yes | - Deploy auto-split toggle to staging and run integration tests (Eng).… |
| `related_incident_id` | 15% | str |  | INC-2026-58 |
| `attachments` | 12% | list[str] | some | /attachments/INV-2026-0331-scanned.pdf |
| `follow_up_actions` | 10% | list[str] | yes | PR: batcher/add-per-traffic-class-gain-control (#4123) |

#### hubspot

15,017 docs; surveyed 2,000. Title field: {'company_name': 1999, 'crm_display_name': 1}.

Content-field variants (most common):

- 11% `['notes']`
- 7% `['notes', 'timeline', 'next_step', 'blockers']`
- 6% `['notes', 'timeline']`
- 5% `['notes', 'next_step', 'blockers']`
- 3% `['notes', 'next_step', 'blockers', 'timeline']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `company_id` | 100% | str |  | CID-2025-0147 |
| `company_name` | 100% | str |  | Cascade Clinical AI |
| `company_domain` | 100% | str |  | cascadeclinical.ai |
| `stage` | 100% | str | some | procurement |
| `owner` | 100% | str |  | Jamie Ortega |
| `created_at` | 100% | str |  | 2025-11-05 |
| `updated_at` | 100% | str |  | 2026-02-28 |
| `account_tier` | 100% | str | some | enterprise |
| `industry` | 100% | str | some | healthcare |
| `employee_count_range` | 100% | str | some | 201-1000 |
| `hq_region` | 100% | str | some | na |
| `interested_products` | 100% | list[str] | some | private |
| `use_cases` | 100% | list[str] | some | support-agent |
| `deployment_requirements` | 100% | list[str] | some | vpc |
| `competitors` | 100% | list[str] | some | Anthropic |
| `last_activity_at` | 100% | str |  | 2026-02-28 |
| `next_step` | 100% | str | yes | Closed-Lost: security evidence not provided. Re-engage only if third-p… |
| `blockers` | 100% | list[str] | yes | security_review_failed |
| `security_requirements` | 100% | list[str] | some | sso_saml |
| `estimated_arr_range` | 100% | str | some | 200k-500k |
| `linked_gmail_threads` | 100% | list[str] |  | thread:1752394829 |
| `linked_drive_docs` | 100% | list[str] |  | https://drive.google.com/pricing_cascade_clinical |
| `linked_fireflies` | 99% | list[str] |  | FF-2025-11-12-1423 |
| `se_assigned` | 99% | str |  | Priya Shah |
| `linked_support_tickets` | 98% | list[str] |  | INF-4321 |
| `notes` | 97% | str | yes | Status: Closed-Lost (security). - Use-case: clinical triage + internal… |
| `forecast_close_month` | 94% | str | some | 2026-01 |
| `csm_assigned` | 50% | str |  | Liam Chen |
| `timeline` | 50% | list[str] | yes | 2025-11-05 - Intro discovery call (AE: Jamie). Identified core need: o… |
| `recent_activities` | 20% | list[str] | yes | 2026-03-02: Intro discovery call (Maya) — product fit looks promising |

#### google_drive

25,108 docs; surveyed 2,000. Title field: {'title': 1997, 'Title': 3}.

Content-field variants (most common):

- 70% `['content']`
- 19% `['body']`
- 1% `['summary', 'content']`
- 1% `['summary', 'body']`
- 0% `['content', 'summary']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `title` | 100% | str |  | DayZero capacity tiling and demand-priced matrix |
| `owner` | 100% | str |  | Aisha Patel |
| `drive_area` | 100% | str |  | shared_drives |
| `path` | 100% | str |  | shared_drives/finance-and-legal/dayzero-capacity-tiling-demand-priced-… |
| `doc_type` | 100% | str |  | sheet |
| `created_at` | 100% | str |  | 2026-05-12 |
| `last_modified` | 100% | str |  | 2026-06-03 |
| `collaborators` | 100% | list[str] |  | Marco Lin |
| `status` | 100% | str |  | in_review |
| `tags` | 100% | list[str] |  | budgeting |
| `linked_artifacts` | 100% | list[str] | some | https://confluence.redwood.ai/pages/DAYZERO-tiling-notes |
| `team` | 100% | str |  | finance |
| `content` | 73% | str | yes | Quick personal scratchpad for the month-end finance/legal items. Use t… |
| `body` | 22% | str | yes | Summary: This sheet captures a compact, deployable approach for aligni… |

#### fireflies

10,173 docs; surveyed 2,000. Title field: {'title': 2000}.

Content-field variants (most common):

- 35% `['transcript']`
- 22% `['summary', 'transcript']`
- 7% `['summary', 'topics', 'action_items', 'next_steps', 'transcript']`
- 6% `['summary', 'topics', 'next_steps', 'action_items', 'transcript']`
- 4% `['summary', 'topics', 'transcript', 'next_steps', 'action_items']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `meeting_id` | 100% | str |  | ff-20251219-novus-3267 |
| `recorded_at` | 100% | str |  | 2025-12-19T13:30:00-08:00 |
| `duration_minutes` | 100% | str |  | 52 |
| `call_type` | 100% | str |  | poc_checkin |
| `title` | 100% | str |  | Novus Health weekly lab bench audit - Redwood vs baseline |
| `redwood_owner` | 100% | str |  | John Carter |
| `redwood_attendees` | 100% | list[str] |  | John Carter (AE) |
| `customer_company` | 100% | str |  | Novus Health |
| `next_steps` | 100% | list[str] | some | Redwood to run 95th percentile latency pass with 1M tokens by 2025-12-… |
| `transcription_quality` | 100% | str |  | medium |
| `customer_attendees` | 100% | list[str] |  | Maria Alvarez (CTO) |
| `competitors_mentioned` | 100% | list[str] |  | CloudServe AI |
| `transcript` | 99% | str | yes | Header: Date: 2025-12-19 Time: 13:30 PST Duration: 52m Attendees: John… |
| `action_items` | 99% | list[str] | some | John Carter: upload perf workbook and annotated runbook; due 2025-12-1… |
| `crm_deal_id` | 86% | str |  | CRM-AC-2028-0614 |
| `summary` | 60% | str | yes | Weekly POC sync focused on recent benchmark runs comparing Redwood inf… |
| `topics` | 59% | list[str] | some | latency vs cost tradeoffs |

#### github

8,052 docs; surveyed 2,000. Title field: {'title': 1999, 'title_short': 1}.

Content-field variants (most common):

- 12% `['description', 'review_comments', 'release_notes']`
- 9% `['description', 'review_comments']`
- 6% `['description', 'release_notes', 'review_comments']`
- 6% `['description', 'release_notes']`
- 4% `['description', 'review_conversation', 'release_notes']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `repo` | 100% | str |  | redwood |
| `pr_number` | 100% | str |  | 128901 |
| `title` | 100% | str |  | Burst sampler heuristics and quant-profile assembler for conservative … |
| `author` | 100% | str |  | Samira Patel |
| `created_at` | 100% | str |  | 2026-02-14 |
| `updated_at` | 100% | str |  | 2026-02-21 |
| `state` | 100% | str |  | merged |
| `base_branch` | 100% | str |  | main |
| `head_branch` | 100% | str |  | feature/econ-burst-sampler-quant-assembler |
| `reviewers` | 100% | list[str] |  | Avery Chen |
| `labels` | 100% | list[str] |  | runtime |
| `linked_linear` | 100% | list[str] |  | ENG-1482 |
| `additions` | 100% | str |  | 612 |
| `deletions` | 100% | str |  | 178 |
| `files_changed_count` | 100% | str |  | 7 |
| `ci_status` | 100% | str |  | pass |
| `merged_at` | 96% | str |  | 2026-02-21 |
| `release_notes` | 96% | str | yes | Introduce burst-aware prefill sampler and quant-profile assembler. Ope… |
| `breaking_change` | 94% | str |  | false |
| `description` | 90% | str | yes | Motivation: reduce unit-token cost volatility during short high-QPS bu… |
| `commits` | 80% | list[str] | some | b8d2f9c: runtime: add BurstSampler with exponential decay window and e… |
| `linked_jira` | 71% | list[str] |  | INT-883 |
| `merge_method` | 67% | str |  | squash |
| `review_comments` | 55% | list[str] | yes | Avery Chen: Nice approach. Can you clarify how the decay window size i… |
| `ci_checks` | 52% | list[str] | some | unit: pass |
| `changed_files` | 48% | list[str] | some | cmd/audit-exporter/main.go (wire config sync manager + flags) |
| `files_changed` | 36% | list[str] | some | src/runtime/prefill_controller.go |
| `merge_outcome` | 29% | str | some | merged |
| `review_conversation` | 15% | list[str] | yes | Diego Morales (reviewer): Can we avoid scanning raw request logs in th… |
| `notes` | 14% | str | yes | Feature is behind config and default sampling is conservative. SRE to … |
| `notes_for_ops` | 11% | str | yes | Enable feature flags: operator.features.consents=false by default. To … |
| `merge_commit` | 10% | str |  | a6b3e1f9d0b14c1e0d6d5b2c9f4e9c2a3d8f71aa |

#### confluence

5,189 docs; surveyed 2,000. Title field: {'title': 1999, 'content': 1}.

Content-field variants (most common):

- 54% `['content']`
- 42% `['body']`
- 1% `['summary', 'body']`
- 1% `['summary', 'content']`
- 0% `['content', 'content_continued']`

| key | fill | type | content? | example |
|---|---|---|---|---|
| `title` | 100% | str |  | ADR-017: Retention policy evaluation in control plane |
| `space` | 100% | str |  | architecture-and-standards |
| `author` | 100% | str |  | Ethan Park |
| `owner_team` | 100% | str |  | engineering |
| `status` | 100% | str |  | published |
| `created_at` | 100% | str |  | 2025-01-22 |
| `last_updated` | 100% | str |  | 2025-02-11 |
| `labels` | 100% | list[str] |  | adr |
| `reviewers` | 100% | list[str] |  | Priya Natarajan |
| `confidentiality` | 99% | str |  | internal |
| `related_pages` | 94% | list[str] | some | sources/confluence/architecture-and-standards/infra-standards/data-del… |
| `content` | 55% | str | yes | ## Status Published (ADR-017) ## Decision summary Retention policies a… |
| `body` | 44% | str | yes | Overview: This playbook defines the Adaptive Eval Funnel (AEF) — a sta… |
